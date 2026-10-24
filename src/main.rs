//! macbench — a small, dependency-free CPU / memory / SSD benchmark for Apple Silicon Macs.
//!
//! Run on each Mac:        cargo run --release -- --save m1pro.tsv
//! Compare result files:   cargo run --release -- compare m1pro.tsv m2pro.tsv
//! Show temperatures:      cargo run --release -- sensors [--watch]
//! Show SSD health:        cargo run --release -- health [diskN | path]

mod health;
mod sensors;

use sensors::{by_category, pressure_label, thermal_pressure, Monitor, Sensors, Summary};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const GIB: usize = 1024 * MIB;
const PAGE: usize = 16 * KIB; // Apple Silicon page size
const DISK_CHUNK: usize = 8 * MIB; // sequential I/O size
const DISK_BLOCK: usize = 4 * KIB; // random I/O size

const USAGE: &str = "\
macbench - CPU / memory / SSD benchmark

USAGE:
    macbench [OPTIONS]
    macbench compare <a.tsv> <b.tsv>
    macbench sensors [--watch]
    macbench health [diskN | path]

OPTIONS:
    --quick            shorter run (~30 s instead of ~2 min)
    --only <list>      comma-separated subset of: cpu,mem,disk
    --threads <n>      threads for multi-threaded tests (default: all cores)
    --dir <path>       directory for the disk test file (default: system temp dir)
    --disk-gb <n>      size of the disk test file in GiB (default: 4, quick: 1)
    --cycles <n>       sequential write/read repetitions (default: 5, quick: 3)
    --no-temps         don't read temperature sensors
    --save <file>      write results to a TSV file for `compare`

SENSORS:
    lists every temperature sensor found; --watch prints a summary every second

HEALTH:
    SSD life used, data written, spare blocks, errors: for every drive, or the one
    holding a path (e.g. /Volumes/External) or named diskN. Readable for the
    internal SSD and Thunderbolt / USB4 NVMe drives; not over USB 3.x.
";

fn usage() -> ! {
    eprint!("{USAGE}");
    process::exit(2)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct Config {
    quick: bool,
    run_cpu: bool,
    run_mem: bool,
    run_disk: bool,
    threads: usize,
    dir: PathBuf,
    disk_bytes: u64,
    cycles: usize,
    temps: bool,
    save: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Config {
    let mut cfg = Config {
        quick: false,
        run_cpu: true,
        run_mem: true,
        run_disk: true,
        threads: thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        dir: env::temp_dir(),
        disk_bytes: 0,
        cycles: 0,
        temps: true,
        save: None,
    };
    let mut only: Option<String> = None;
    let mut disk_gb: Option<f64> = None;
    let mut cycles: Option<usize> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--quick" => cfg.quick = true,
            "--only" => only = Some(it.next().unwrap_or_else(|| usage()).clone()),
            "--threads" => {
                cfg.threads = it
                    .next()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or_else(|| usage())
            }
            "--dir" => cfg.dir = PathBuf::from(it.next().unwrap_or_else(|| usage())),
            "--disk-gb" => {
                disk_gb = Some(
                    it.next()
                        .and_then(|v| v.parse::<f64>().ok())
                        .filter(|&g| g > 0.0)
                        .unwrap_or_else(|| usage()),
                )
            }
            "--cycles" => {
                cycles = Some(
                    it.next()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or_else(|| usage()),
                )
            }
            "--no-temps" => cfg.temps = false,
            "--save" => cfg.save = Some(PathBuf::from(it.next().unwrap_or_else(|| usage()))),
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument: {other}\n");
                usage()
            }
        }
    }

    if let Some(list) = only {
        cfg.run_cpu = false;
        cfg.run_mem = false;
        cfg.run_disk = false;
        for part in list.split(',') {
            match part.trim() {
                "cpu" => cfg.run_cpu = true,
                "mem" => cfg.run_mem = true,
                "disk" => cfg.run_disk = true,
                _ => usage(),
            }
        }
    }

    let gb = disk_gb.unwrap_or(if cfg.quick { 1.0 } else { 4.0 });
    let bytes = (gb * GIB as f64) as u64;
    cfg.disk_bytes = (bytes / DISK_CHUNK as u64).max(1) * DISK_CHUNK as u64;
    cfg.cycles = cycles.unwrap_or(if cfg.quick { 3 } else { 5 });
    cfg
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

/// splitmix64 — fast, good-enough PRNG with no dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x5DEE_CE66_D1CE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, n).
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
    /// Uniform in [0, 1).
    fn unit_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Page-aligned buffer (uncached I/O on macOS is happiest with aligned buffers).
struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, PAGE).expect("bad layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedBuf { ptr, layout }
    }
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    const F_NOCACHE: i32 = 48;

    /// Bypass the unified buffer cache so we measure the SSD, not RAM.
    pub fn no_cache(f: &File) {
        unsafe {
            fcntl(f.as_raw_fd(), F_NOCACHE, 1i32);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod mac {
    use std::fs::File;
    pub fn no_cache(_f: &File) {}
}

fn best_of<R>(runs: usize, mut f: impl FnMut() -> R) -> Duration {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed()
        })
        .min()
        .unwrap()
}

fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

fn fmt_num(v: f64) -> String {
    if v >= 1000.0 {
        let s = format!("{v:.0}");
        let mut out = String::new();
        for (i, ch) in s.chars().enumerate() {
            if i > 0 && (s.len() - i) % 3 == 0 {
                out.push(',');
            }
            out.push(ch);
        }
        out
    } else if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

fn section(title: &str) {
    println!("\n== {title} ==");
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

struct Row {
    key: String,
    value: f64,
    unit: String,
    higher_better: bool,
}

#[derive(Default)]
struct Results {
    rows: Vec<Row>,
}

impl Results {
    /// Record and print a result.
    fn add(&mut self, key: impl Into<String>, value: f64, unit: &str, higher_better: bool) {
        let key = key.into();
        println!("  {:<24} {:>12} {}", key, fmt_num(value), unit);
        self.record(key, value, unit, higher_better);
    }

    /// Record a result for `--save` without printing it.
    fn record(&mut self, key: impl Into<String>, value: f64, unit: &str, higher_better: bool) {
        self.rows.push(Row { key: key.into(), value, unit: unit.to_string(), higher_better });
    }
}

// ---------------------------------------------------------------------------
// Temperatures
// ---------------------------------------------------------------------------

const TEMP_UNIT: &str = "°C";
const PRESSURE_UNIT: &str = "level";

/// "CPU P-cores 41°C · SSD 37°C · pressure nominal", or None if nothing is readable.
fn temps_line(sensors: Option<&Sensors>) -> Option<String> {
    let mut parts: Vec<String> = sensors
        .map(|s| by_category(&s.read()))
        .unwrap_or_default()
        .into_iter()
        .map(|(cat, t)| format!("{cat} {t:.0}°C"))
        .collect();
    if let Some(p) = thermal_pressure() {
        parts.push(format!("pressure {}", pressure_label(p)));
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// Runs one benchmark section while sampling temperatures in the background,
/// then prints start → end / peak per sensor group and saves the peaks.
fn with_temps(cfg: &Config, res: &mut Results, section: &str, run: impl FnOnce(&mut Results)) {
    let monitor = cfg.temps.then(|| Monitor::start(Duration::from_millis(500)));
    run(res);
    if let Some(m) = monitor {
        report_temps(section, &m.finish(), res);
    }
}

fn report_temps(section: &str, s: &Summary, res: &mut Results) {
    if s.is_empty() {
        return;
    }
    println!("  temperatures (start → end, peak):");
    for (cat, span) in &s.spans {
        println!(
            "    {:<16} {:>4.0} → {:>3.0} °C   peak {:.0} °C",
            cat, span.first, span.last, span.peak
        );
    }
    if let Some(p) = s.worst_pressure {
        println!("    {:<16} {} (worst seen)", "thermal pressure", pressure_label(p));
        res.record(format!("{section}.thermal_pressure"), p as f64, PRESSURE_UNIT, false);
    }
    if let Some(t) = s.peak_where(|c| c.starts_with("CPU")) {
        res.record(format!("{section}.temp.cpu_peak"), t, TEMP_UNIT, false);
    }
    if section == "disk" {
        if let Some(t) = s.peak_where(|c| c == sensors::INTERNAL_SSD) {
            res.record("disk.temp.ssd_peak", t, TEMP_UNIT, false);
        }
    }
}

fn sensors_cmd(args: &[String]) {
    let watch = match args {
        [] => false,
        [w] if w == "--watch" => true,
        _ => usage(),
    };
    let sensors = Sensors::open();
    if sensors.is_none() {
        println!("no temperature sensors found (they're read on macOS / Apple Silicon only)");
    }

    if !watch {
        if let Some(s) = &sensors {
            let mut readings = s.read();
            readings.sort_by(|a, b| a.name.cmp(&b.name));
            println!("{:<32} {:>6}  group", "sensor", "°C");
            for r in &readings {
                let group = sensors::category(&r.name).unwrap_or("-");
                println!("{:<32} {:>6.1}  {}", r.name, r.celsius, group);
            }
            println!("\n{} sensors", readings.len());
        }
        if let Some(p) = thermal_pressure() {
            println!("thermal pressure: {}", pressure_label(p));
        }
        return;
    }

    let start = Instant::now();
    loop {
        let line = temps_line(sensors.as_ref()).unwrap_or_else(|| "nothing readable".into());
        println!("[{:>5.0}s]  {line}", start.elapsed().as_secs_f64());
        thread::sleep(Duration::from_secs(1));
    }
}

// ---------------------------------------------------------------------------
// SSD health
// ---------------------------------------------------------------------------

const LIFE_UNIT: &str = "% life used";
const WRITTEN_UNIT: &str = "TB written";

/// Decimal units, like drive makers and Finder use.
fn fmt_bytes(bytes: f64) -> String {
    match bytes {
        b if b >= 1e12 => format!("{:.2} TB", b / 1e12),
        b if b >= 1e9 => format!("{:.1} GB", b / 1e9),
        b => format!("{:.0} MB", b / 1e6),
    }
}

/// "APPLE SSD AP0512Z · disk0 · Apple Fabric, internal · 500.3 GB · 1% life used · 12.34 TB written"
fn drive_summary(d: &health::Drive) -> String {
    let mut s = format!(
        "{} · {} · {}, {} · {}",
        d.model,
        d.bsd,
        d.interconnect,
        d.location.to_lowercase(),
        fmt_bytes(d.size as f64)
    );
    match &d.health {
        Ok(h) => s += &format!(" · {}% life used · {} written", h.percent_used, fmt_bytes(h.bytes_written as f64)),
        Err(_) => s += " · health not readable",
    }
    s
}

fn print_drive(d: &health::Drive) {
    let also = if d.aliases.is_empty() { String::new() } else { format!("  (also {})", d.aliases.join(", ")) };
    println!("{}  {}{}", d.bsd, d.model, also);
    println!(
        "  {:<14} {}, {} · {}",
        "connection",
        d.interconnect,
        d.location.to_lowercase(),
        fmt_bytes(d.size as f64)
    );
    let h = match &d.health {
        Ok(h) => h,
        Err(why) => {
            println!("  {:<14} {why}", "health");
            return;
        }
    };
    let warnings = h.warnings();
    let status = if warnings.is_empty() { "OK, no warnings".to_string() } else { format!("WARNING: {}", warnings.join("; ")) };
    println!("  {:<14} {status}", "status");
    println!("  {:<14} {} %   (controller's estimate of rated write endurance used)", "life used", h.percent_used);
    println!("  {:<14} {} %   (warns below {} %)", "spare blocks", h.available_spare, h.spare_threshold);
    println!("  {:<14} {}", "data written", fmt_bytes(h.bytes_written as f64));
    println!("  {:<14} {}", "data read", fmt_bytes(h.bytes_read as f64));
    println!(
        "  {:<14} {} h · {} power cycles · {} unsafe shutdowns",
        "power on",
        fmt_num(h.power_on_hours as f64),
        fmt_num(h.power_cycles as f64),
        fmt_num(h.unsafe_shutdowns as f64)
    );
    if let Some(t) = h.temperature_c {
        println!("  {:<14} {t:.0} °C", "temperature");
    }
    println!("  {:<14} {} media errors · {} error log entries", "errors", h.media_errors, h.error_log_entries);
}

fn health_cmd(args: &[String]) {
    let drives = match args {
        [] => health::list(),
        [target] => match health::find(target) {
            Some(d) => vec![d],
            None => {
                eprintln!("no physical drive found for {target}");
                process::exit(1);
            }
        },
        _ => usage(),
    };
    if drives.is_empty() {
        if cfg!(target_os = "macos") {
            println!("no drives found; try `macbench health disk0`, or a path such as /");
        } else {
            println!("no drives found (drive health is read on macOS only)");
        }
    }
    for (i, d) in drives.iter().enumerate() {
        if i > 0 {
            println!();
        }
        print_drive(d);
    }
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------

fn cmd(prog: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(prog).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn sysctl(name: &str) -> Option<String> {
    cmd("sysctl", &["-n", name])
}

fn machine_name() -> String {
    let cpu = sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown CPU".into());
    match sysctl("hw.model") {
        Some(model) => format!("{cpu} ({model})"),
        None => cpu,
    }
}

fn print_system_info(cfg: &Config) {
    println!("macbench {}", env!("CARGO_PKG_VERSION"));
    println!("  machine  {}", machine_name());
    if let (Some(p), Some(e)) = (sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu")) {
        println!("  cores    {p} performance + {e} efficiency");
    }
    if let Some(mem) = sysctl("hw.memsize").and_then(|m| m.parse::<u64>().ok()) {
        println!("  memory   {} GB", mem / (1 << 30));
    }
    if let Some(v) = cmd("sw_vers", &["-productVersion"]) {
        println!("  macOS    {v}");
    }
    println!("  threads  {} for multi-threaded tests", cfg.threads);
    if cfg.temps {
        if let Some(t) = temps_line(Sensors::open().as_ref()) {
            println!("  temps    {t}");
        }
    }
    if cmd("pmset", &["-g", "batt"]).is_some_and(|s| s.contains("Battery Power")) {
        println!("\n  ! running on battery: plug in for consistent results");
    }
}

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------

/// One "op" of a CPU workload. Each takes roughly 5–30 ms on an M-series P-core.
type Unit = fn(u64) -> u64;

const CPU_TESTS: [(&str, Unit); 4] = [
    ("sieve", unit_sieve as Unit),           // integer + branchy, cache-heavy
    ("sort", unit_sort as Unit),             // branch prediction, memory
    ("mandelbrot", unit_mandelbrot as Unit), // FP latency chains
    ("matmul", unit_matmul as Unit),         // FP throughput, SIMD
];

fn unit_sieve(_seed: u64) -> u64 {
    const N: usize = 4_000_000;
    let mut composite = vec![false; N + 1];
    let mut count = 0u64;
    for i in 2..=N {
        if !composite[i] {
            count += 1;
            let mut j = i * i;
            while j <= N {
                composite[j] = true;
                j += i;
            }
        }
    }
    count
}

fn unit_sort(seed: u64) -> u64 {
    let mut rng = Rng::new(seed);
    let mut v: Vec<u64> = (0..1_000_000).map(|_| rng.next()).collect();
    v.sort_unstable();
    v[v.len() / 2]
}

fn unit_mandelbrot(seed: u64) -> u64 {
    const W: usize = 512;
    const H: usize = 512;
    const MAX_ITER: u32 = 256;
    let jitter = (seed % 7) as f64 * 1e-9;
    let mut total = 0u64;
    for py in 0..H {
        let ci = -1.2 + 2.4 * py as f64 / H as f64 + jitter;
        for px in 0..W {
            let cr = -2.2 + 3.0 * px as f64 / W as f64;
            let (mut zr, mut zi) = (0.0f64, 0.0f64);
            let mut n = 0;
            while n < MAX_ITER && zr * zr + zi * zi <= 4.0 {
                let t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
                n += 1;
            }
            total += n as u64;
        }
    }
    total
}

fn unit_matmul(seed: u64) -> u64 {
    const N: usize = 384; // multiple of 8
    let mut rng = Rng::new(seed);
    let a: Vec<f64> = (0..N * N).map(|_| rng.unit_f64()).collect();
    let b: Vec<f64> = (0..N * N).map(|_| rng.unit_f64()).collect();
    let mut bt = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            bt[j * N + i] = b[i * N + j];
        }
    }
    let mut c = vec![0.0f64; N * N];
    for i in 0..N {
        let row = &a[i * N..(i + 1) * N];
        for j in 0..N {
            let col = &bt[j * N..(j + 1) * N];
            // 8 independent accumulators so LLVM can vectorize the dot product.
            let mut s = [0.0f64; 8];
            for (r, k) in row.chunks_exact(8).zip(col.chunks_exact(8)) {
                for l in 0..8 {
                    s[l] += r[l] * k[l];
                }
            }
            c[i * N + j] = s.iter().sum();
        }
    }
    c.iter().sum::<f64>().to_bits()
}

/// Runs `total` ops spread across `threads` threads. Threads pull work from a
/// shared counter, so P-cores and E-cores each contribute what they can.
fn run_units(unit: Unit, total: usize, threads: usize) -> u64 {
    let next = AtomicUsize::new(0);
    let acc = AtomicU64::new(0);
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut local = 0u64;
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= total {
                        break;
                    }
                    local = local.wrapping_add(unit(i as u64));
                }
                acc.fetch_add(local, Relaxed);
            });
        }
    });
    acc.load(Relaxed)
}

fn run_cpu(cfg: &Config, res: &mut Results) {
    section("CPU  (ops/s, higher is better)");
    let target = if cfg.quick { 0.4 } else { 1.5 }; // seconds per single-thread measurement
    let runs = if cfg.quick { 2 } else { 3 };
    let (mut st_all, mut mt_all) = (Vec::new(), Vec::new());

    for (name, unit) in CPU_TESTS {
        black_box(unit(0)); // warm-up
        let t = Instant::now();
        black_box(unit(1));
        let one = t.elapsed().as_secs_f64().max(1e-6);
        let reps = ((target / one).ceil() as usize).max(2);

        let st = best_of(runs, || run_units(unit, reps, 1));
        let mt_total = reps * cfg.threads;
        let mt = best_of(runs, || run_units(unit, mt_total, cfg.threads));

        let st_rate = reps as f64 / st.as_secs_f64();
        let mt_rate = mt_total as f64 / mt.as_secs_f64();
        res.add(format!("cpu.{name}.st"), st_rate, "ops/s", true);
        res.add(format!("cpu.{name}.mt"), mt_rate, "ops/s", true);
        st_all.push(st_rate);
        mt_all.push(mt_rate);
    }

    let (st, mt) = (geomean(&st_all), geomean(&mt_all));
    res.add("cpu.score.st", st, "pts (geomean)", true);
    res.add("cpu.score.mt", mt, "pts (geomean)", true);
    res.add("cpu.scaling", mt / st, "x", true);
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

fn split(len: usize, threads: usize) -> usize {
    len.div_ceil(threads).max(1)
}

fn par_read(data: &[u64], threads: usize, passes: usize) -> u64 {
    thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(split(data.len(), threads))
            .map(|c| {
                s.spawn(move || {
                    let mut acc = 0u64;
                    for _ in 0..passes {
                        let sum = black_box(c).iter().fold(0u64, |a, &x| a.wrapping_add(x));
                        acc = acc.wrapping_add(sum);
                    }
                    acc
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).fold(0, u64::wrapping_add)
    })
}

fn par_write(data: &mut [u64], threads: usize, passes: usize, v: u64) {
    let chunk = split(data.len(), threads);
    thread::scope(|s| {
        for c in data.chunks_mut(chunk) {
            s.spawn(move || {
                for p in 0..passes {
                    c.fill(v.wrapping_add(p as u64));
                    black_box(&mut *c);
                }
            });
        }
    });
}

fn par_copy(src: &[u64], dst: &mut [u64], threads: usize, passes: usize) {
    let chunk = split(src.len(), threads);
    thread::scope(|s| {
        for (a, b) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            s.spawn(move || {
                for _ in 0..passes {
                    b.copy_from_slice(black_box(a));
                    black_box(&mut *b);
                }
            });
        }
    });
}

/// Average latency of a dependent random load, via pointer chasing through a
/// single random cycle (Sattolo's algorithm) that spans `bytes` of memory.
fn chase_latency(bytes: usize, steps: usize) -> f64 {
    let n = bytes / std::mem::size_of::<usize>();
    let mut next: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(bytes as u64);
    for i in (1..n).rev() {
        let j = rng.below(i as u64) as usize;
        next.swap(i, j);
    }
    let mut p = 0usize;
    for _ in 0..n.min(steps) {
        p = next[p]; // warm caches and TLB
    }
    let t = Instant::now();
    for _ in 0..steps {
        // SAFETY: every element of `next` is < n.
        p = unsafe { *next.get_unchecked(p) };
    }
    let ns = t.elapsed().as_nanos() as f64 / steps as f64;
    black_box(p);
    ns
}

fn run_mem(cfg: &Config, res: &mut Results) {
    section("Memory  (GB/s higher is better; copy counts read + write; latency ns lower is better)");
    let bytes = if cfg.quick { 256 * MIB } else { GIB };
    let n = bytes / 8;
    let runs = if cfg.quick { 3 } else { 5 };

    // Both buffers are written here so every page is really mapped before timing.
    let src: Vec<u64> = (0..n as u64).collect();
    let mut dst = vec![1u64; n];

    let mut modes = vec![(1usize, "st")];
    if cfg.threads > 1 {
        modes.push((cfg.threads, "mt"));
    }
    for (threads, tag) in modes {
        let passes = if threads == 1 { 2 } else { 8 };
        let gb = (bytes * passes) as f64 / 1e9;

        let t = best_of(runs, || par_read(&src, threads, passes));
        res.add(format!("mem.read.{tag}"), gb / t.as_secs_f64(), "GB/s", true);

        let mut v = 0u64;
        let t = best_of(runs, || {
            v += 1;
            par_write(&mut dst, threads, passes, v)
        });
        res.add(format!("mem.write.{tag}"), gb / t.as_secs_f64(), "GB/s", true);

        let t = best_of(runs, || par_copy(&src, &mut dst, threads, passes));
        res.add(format!("mem.copy.{tag}"), 2.0 * gb / t.as_secs_f64(), "GB/s", true);
    }
    drop(src);
    drop(dst);

    // Sizes chosen to land in L1, L2, around the L2/SLC boundary, and DRAM.
    let steps = if cfg.quick { 2_000_000 } else { 10_000_000 };
    for (size, label) in [
        (32 * KIB, "32KiB"),
        (256 * KIB, "256KiB"),
        (4 * MIB, "4MiB"),
        (16 * MIB, "16MiB"),
        (64 * MIB, "64MiB"),
        (256 * MIB, "256MiB"),
    ] {
        res.add(format!("mem.latency.{label}"), chase_latency(size, steps), "ns", false);
    }
}

// ---------------------------------------------------------------------------
// Disk
// ---------------------------------------------------------------------------

/// Deletes the test file even if a test fails part-way.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn rand_read(path: &Path, file_len: u64, threads: usize, dur: Duration) -> io::Result<f64> {
    let blocks = file_len / DISK_BLOCK as u64;
    let stop = AtomicBool::new(false);
    let ops = AtomicU64::new(0);
    let t = Instant::now();
    thread::scope(|s| -> io::Result<()> {
        let handles: Vec<_> = (0..threads)
            .map(|id| {
                let (stop, ops) = (&stop, &ops);
                s.spawn(move || -> io::Result<()> {
                    let f = File::open(path)?;
                    mac::no_cache(&f);
                    let mut buf = AlignedBuf::new(DISK_BLOCK);
                    let mut rng = Rng::new(1000 + id as u64);
                    let mut n = 0u64;
                    while !stop.load(Relaxed) {
                        let off = rng.below(blocks) * DISK_BLOCK as u64;
                        f.read_exact_at(buf.as_mut_slice(), off)?;
                        n += 1;
                    }
                    ops.fetch_add(n, Relaxed);
                    Ok(())
                })
            })
            .collect();
        thread::sleep(dur);
        stop.store(true, Relaxed);
        for h in handles {
            h.join().expect("disk thread panicked")?;
        }
        Ok(())
    })?;
    Ok(ops.load(Relaxed) as f64 / t.elapsed().as_secs_f64())
}

/// Prints a row of `DOTS` dots as a sequential pass progresses, like ssd-benchmark.
struct Progress {
    chunks: u64,
    printed: u64,
}

impl Progress {
    const DOTS: u64 = 20;

    fn new(chunks: u64) -> Self {
        Progress { chunks: chunks.max(1), printed: 0 }
    }
    fn tick(&mut self, done: u64) {
        let due = done * Self::DOTS / self.chunks;
        if due > self.printed {
            print!("{}", ".".repeat((due - self.printed) as usize));
            let _ = io::stdout().flush();
            self.printed = due;
        }
    }
}

/// One pass of uncached 8 MiB writes over the whole file, timed through a full
/// flush (std's sync_all uses F_FULLFSYNC on macOS, so the data really hits NAND).
fn seq_write(path: &Path, total: u64, buf: &AlignedBuf) -> io::Result<f64> {
    let chunks = total / DISK_CHUNK as u64;
    let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
    mac::no_cache(&f);
    let mut progress = Progress::new(chunks);
    let t = Instant::now();
    for i in 1..=chunks {
        f.write_all(buf.as_slice())?;
        progress.tick(i);
    }
    f.sync_all()?;
    Ok(total as f64 / 1e9 / t.elapsed().as_secs_f64())
}

/// One pass of uncached 8 MiB reads over the whole file.
fn seq_read(path: &Path, total: u64, buf: &mut AlignedBuf) -> io::Result<f64> {
    let chunks = total / DISK_CHUNK as u64;
    let mut f = File::open(path)?;
    mac::no_cache(&f);
    let mut progress = Progress::new(chunks);
    let t = Instant::now();
    for i in 1..=chunks {
        f.read_exact(buf.as_mut_slice())?;
        progress.tick(i);
    }
    Ok(total as f64 / 1e9 / t.elapsed().as_secs_f64())
}

/// Repeats a sequential pass `n` times, printing each cycle's speed and the SSD
/// temperature after it, then records mean and worst cycle and prints the spread.
fn cycles(
    res: &mut Results,
    key: &str,
    label: &str,
    n: usize,
    ssd_temp: &dyn Fn() -> Option<f64>,
    mut pass: impl FnMut() -> io::Result<f64>,
) -> io::Result<()> {
    let mut speeds = Vec::with_capacity(n);
    for i in 1..=n {
        print!("  {label:<10} [{i}/{n}] ");
        let _ = io::stdout().flush();
        let gbps = pass()?;
        print!("  {gbps:>6.2} GB/s");
        if let Some(t) = ssd_temp() {
            print!("   SSD {t:.0} °C");
        }
        println!();
        speeds.push(gbps);
    }

    let mean = speeds.iter().sum::<f64>() / n as f64;
    let sd = (speeds.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let (min, max) = speeds.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    res.add(key, mean, "GB/s  (mean)", true);
    res.add(format!("{key}.worst"), min, "GB/s  (slowest cycle)", true);
    println!(
        "  {:<24} {:>12} GB/s  (fastest cycle; σ {:.2} = {:.1}%)",
        "",
        fmt_num(max),
        sd,
        sd / mean * 100.0
    );
    Ok(())
}

fn run_disk(cfg: &Config, res: &mut Results) -> io::Result<()> {
    let total = cfg.disk_bytes;
    section(&format!(
        "Disk  ({:.1} GiB file in {}, {} cycles; higher is better)",
        total as f64 / GIB as f64,
        cfg.dir.display(),
        cfg.cycles
    ));
    let file = TempFile(cfg.dir.join(format!("macbench-{}.tmp", process::id())));
    let path = file.0.as_path();

    // Random bytes, so nothing along the way can compress or dedupe the data.
    let mut buf = AlignedBuf::new(DISK_CHUNK);
    let mut rng = Rng::new(7);
    for w in buf.as_mut_slice().chunks_exact_mut(8) {
        w.copy_from_slice(&rng.next().to_le_bytes());
    }

    // Which physical drive is under test, and how worn it is.
    let drive = health::find(&cfg.dir.to_string_lossy());
    if let Some(d) = &drive {
        println!("  drive    {}\n", drive_summary(d));
        if let Ok(h) = &d.health {
            res.record("disk.health.life_used", h.percent_used as f64, LIFE_UNIT, false);
            res.record("disk.health.tb_written", h.bytes_written as f64 / 1e12, WRITTEN_UNIT, false);
        }
    }

    // Per-cycle temperature of the drive under test: from its own health log when
    // readable, else the internal NAND sensor — but only if the drive is internal.
    let sensors = if cfg.temps { Sensors::open() } else { None };
    let from_smart = cfg.temps && drive.as_ref().is_some_and(|d| d.health.is_ok());
    let internal = drive.as_ref().is_none_or(|d| d.is_internal());
    let ssd_temp = || {
        if from_smart {
            return drive.as_ref()?.refresh()?.temperature_c;
        }
        let s = sensors.as_ref().filter(|_| internal)?;
        by_category(&s.read()).into_iter().find(|(c, _)| *c == sensors::INTERNAL_SSD).map(|(_, t)| t)
    };

    cycles(res, "disk.seq_write", "seq write", cfg.cycles, &ssd_temp, || seq_write(path, total, &buf))?;
    println!();
    cycles(res, "disk.seq_read", "seq read", cfg.cycles, &ssd_temp, || seq_read(path, total, &mut buf))?;
    println!();

    // Random 4 KiB reads: one thread (QD1 — pure latency) and 16 threads (~QD16).
    let dur = Duration::from_secs(if cfg.quick { 2 } else { 5 });
    for (threads, label) in [(1, "qd1"), (16, "qd16")] {
        let iops = rand_read(path, total, threads, dur)?;
        res.add(format!("disk.rand4k_read.{label}"), iops, "IOPS", true);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Save / compare
// ---------------------------------------------------------------------------

fn save(path: &Path, res: &Results) -> io::Result<()> {
    let mut out = format!("#machine\t{}\n", machine_name());
    for r in &res.rows {
        out += &format!("{}\t{}\t{}\t{}\n", r.key, r.value, r.unit, r.higher_better as u8);
    }
    fs::write(path, out)
}

fn load(path: &Path) -> io::Result<(String, Vec<Row>)> {
    let text = fs::read_to_string(path)?;
    let mut name = path.display().to_string();
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(n) = line.strip_prefix("#machine\t") {
            name = n.to_string();
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            continue;
        }
        let Ok(value) = f[1].parse::<f64>() else { continue };
        rows.push(Row { key: f[0].into(), value, unit: f[2].into(), higher_better: f[3] == "1" });
    }
    Ok((name, rows))
}

fn compare(a: &Path, b: &Path) -> io::Result<()> {
    let (name_a, rows_a) = load(a)?;
    let (name_b, rows_b) = load(b)?;
    println!("A = {name_a}\nB = {name_b}\n");
    println!("{:<24} {:>12} {:>12}  {:>8}", "metric", "A", "B", "B vs A");
    println!("{}", "-".repeat(70));
    for x in &rows_a {
        let Some(y) = rows_b.iter().find(|y| y.key == x.key) else { continue };
        // Temperatures and pressure levels read better as a difference than a ratio.
        let delta = if [TEMP_UNIT, PRESSURE_UNIT, LIFE_UNIT].contains(&x.unit.as_str()) {
            format!("{:>+7.0} {}", y.value - x.value, x.unit)
        } else if x.unit == WRITTEN_UNIT {
            format!("{:>+7.2} {}", y.value - x.value, x.unit)
        } else if x.value > 0.0 && y.value > 0.0 {
            let speedup = if x.higher_better { y.value / x.value } else { x.value / y.value };
            format!("{:>+7.1}%  {}", (speedup - 1.0) * 100.0, x.unit)
        } else {
            format!("{:>8}  {}", "n/a", x.unit)
        };
        println!("{:<24} {:>12} {:>12}  {delta}", x.key, fmt_num(x.value), fmt_num(y.value));
    }
    println!("\nB vs A: positive % means B is better (faster, more throughput, or lower latency).");
    println!("Temperatures, pressure and drive wear show B minus A (negative: B ran cooler / is less worn).");
    Ok(())
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("sensors") => return sensors_cmd(&args[1..]),
        Some("health") => return health_cmd(&args[1..]),
        _ => {}
    }
    if args.first().map(String::as_str) == Some("compare") {
        if args.len() != 3 {
            usage();
        }
        if let Err(e) = compare(Path::new(&args[1]), Path::new(&args[2])) {
            eprintln!("compare failed: {e}");
            process::exit(1);
        }
        return;
    }

    let cfg = parse_args(&args);
    print_system_info(&cfg);
    let mut res = Results::default();

    if cfg.run_cpu {
        with_temps(&cfg, &mut res, "cpu", |res| run_cpu(&cfg, res));
    }
    if cfg.run_mem {
        with_temps(&cfg, &mut res, "mem", |res| run_mem(&cfg, res));
    }
    if cfg.run_disk {
        with_temps(&cfg, &mut res, "disk", |res| {
            if let Err(e) = run_disk(&cfg, res) {
                eprintln!("  disk test failed: {e}");
            }
        });
    }

    if let Some(path) = &cfg.save {
        match save(path, &res) {
            Ok(()) => println!("\nsaved results to {}", path.display()),
            Err(e) => eprintln!("\ncould not save results: {e}"),
        }
    }
}
