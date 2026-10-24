//! Temperature sensors and thermal pressure.
//!
//! Apple Silicon has no public API for die temperatures. On macOS this uses the
//! private IOKit HID event interface that tools like `stats` and `macmon` use; it
//! needs no root. Sensor names differ between chips, so `macbench sensors` lists
//! everything that was found, and `category()` groups the names we recognise.
//!
//! Thermal pressure is the public, coarse signal macOS itself uses to decide when
//! to throttle: nominal → moderate → heavy → trapping → sleeping.

use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub use imp::{thermal_pressure, Sensors};

pub struct Reading {
    pub name: String,
    pub celsius: f64,
}

/// The internal SSD's NAND sensor (external drives report via `health`).
pub const INTERNAL_SSD: &str = "Internal SSD";

/// Display order for sensor groups.
pub const CATEGORIES: [&str; 7] =
    ["CPU P-cores", "CPU E-cores", "CPU die", "GPU", "SoC", INTERNAL_SSD, "Battery"];

/// Best-effort grouping of Apple Silicon sensor names, e.g. "pACC MTR Temp Sensor2",
/// "eACC MTR Temp Sensor0", "PMU tdie3", "GPU MTR Temp Sensor1", "NAND CH0 temp".
pub fn category(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    if n.contains("pacc") {
        Some("CPU P-cores")
    } else if n.contains("eacc") {
        Some("CPU E-cores")
    } else if n.contains("tdie") {
        Some("CPU die")
    } else if n.contains("gpu") {
        Some("GPU")
    } else if n.contains("soc") {
        Some("SoC")
    } else if n.contains("nand") {
        Some(INTERNAL_SSD)
    } else if n.contains("battery") || n.contains("gas gauge") {
        Some("Battery")
    } else {
        None
    }
}

const PRESSURE_LEVELS: [&str; 5] = ["nominal", "moderate", "heavy", "trapping", "sleeping"];

pub fn pressure_label(level: u64) -> &'static str {
    PRESSURE_LEVELS.get(level as usize).copied().unwrap_or("unknown")
}

/// Hottest reading in each recognised group, in `CATEGORIES` order.
pub fn by_category(readings: &[Reading]) -> Vec<(&'static str, f64)> {
    CATEGORIES
        .iter()
        .filter_map(|&cat| {
            readings
                .iter()
                .filter(|r| category(&r.name) == Some(cat))
                .map(|r| r.celsius)
                .reduce(f64::max)
                .map(|t| (cat, t))
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub first: f64,
    pub last: f64,
    pub peak: f64,
}

/// Temperatures seen over a stretch of time (one benchmark section).
#[derive(Default, Debug)]
pub struct Summary {
    pub spans: Vec<(&'static str, Span)>,
    pub worst_pressure: Option<u64>,
}

impl Summary {
    pub fn update(&mut self, readings: &[Reading], pressure: Option<u64>) {
        for (cat, t) in by_category(readings) {
            match self.spans.iter_mut().find(|(c, _)| *c == cat) {
                Some((_, s)) => {
                    s.last = t;
                    s.peak = s.peak.max(t);
                }
                None => self.spans.push((cat, Span { first: t, last: t, peak: t })),
            }
        }
        self.spans.sort_by_key(|(c, _)| CATEGORIES.iter().position(|x| x == c));
        if let Some(p) = pressure {
            self.worst_pressure = Some(self.worst_pressure.map_or(p, |w| w.max(p)));
        }
    }

    /// Highest peak among the groups matching `pred`.
    pub fn peak_where(&self, pred: impl Fn(&str) -> bool) -> Option<f64> {
        self.spans.iter().filter(|(c, _)| pred(c)).map(|(_, s)| s.peak).reduce(f64::max)
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty() && self.worst_pressure.is_none()
    }
}

/// Samples temperatures on a background thread until `finish()` is called.
pub struct Monitor {
    stop: Sender<()>,
    handle: JoinHandle<Summary>,
}

impl Monitor {
    pub fn start(interval: Duration) -> Self {
        let (stop, rx) = mpsc::channel::<()>();
        let handle = thread::spawn(move || {
            // Opened on this thread: the IOKit handles aren't Send.
            let sensors = Sensors::open();
            let sample = |s: &mut Summary| {
                let readings = sensors.as_ref().map(|x| x.read()).unwrap_or_default();
                s.update(&readings, thermal_pressure());
            };
            let mut summary = Summary::default();
            sample(&mut summary);
            while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(interval) {
                sample(&mut summary);
            }
            sample(&mut summary);
            summary
        });
        Monitor { stop, handle }
    }

    pub fn finish(self) -> Summary {
        drop(self.stop);
        self.handle.join().unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// macOS implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod imp {
    use super::Reading;
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr::addr_of;

    type Ref = *const c_void;

    #[repr(C)]
    struct Opaque {
        _p: [u8; 0],
    }

    const UTF8: u32 = 0x0800_0100; // kCFStringEncodingUTF8
    const SINT32: isize = 3; // kCFNumberSInt32Type
    const EVENT_TEMPERATURE: i64 = 15; // kIOHIDEventTypeTemperature

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFAllocatorDefault: Ref;
        static kCFTypeDictionaryKeyCallBacks: Opaque;
        static kCFTypeDictionaryValueCallBacks: Opaque;
        fn CFStringCreateWithCString(alloc: Ref, s: *const c_char, enc: u32) -> Ref;
        fn CFStringGetCString(s: Ref, buf: *mut c_char, size: isize, enc: u32) -> u8;
        fn CFStringGetTypeID() -> usize;
        fn CFGetTypeID(cf: Ref) -> usize;
        fn CFNumberCreate(alloc: Ref, kind: isize, value: *const c_void) -> Ref;
        fn CFDictionaryCreate(
            alloc: Ref,
            keys: *const Ref,
            values: *const Ref,
            n: isize,
            key_cb: *const Opaque,
            value_cb: *const Opaque,
        ) -> Ref;
        fn CFArrayGetCount(a: Ref) -> isize;
        fn CFArrayGetValueAtIndex(a: Ref, i: isize) -> Ref;
        fn CFRelease(cf: Ref);
    }

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOHIDEventSystemClientCreate(alloc: Ref) -> Ref;
        fn IOHIDEventSystemClientSetMatching(client: Ref, matching: Ref) -> i32;
        fn IOHIDEventSystemClientCopyServices(client: Ref) -> Ref;
        fn IOHIDServiceClientCopyProperty(service: Ref, key: Ref) -> Ref;
        fn IOHIDServiceClientCopyEvent(service: Ref, kind: i64, options: i32, timestamp: i64) -> Ref;
        fn IOHIDEventGetFloatValue(event: Ref, field: u32) -> f64;
    }

    // libSystem (notify.h)
    extern "C" {
        fn notify_register_check(name: *const c_char, token: *mut i32) -> u32;
        fn notify_get_state(token: i32, state: *mut u64) -> u32;
        fn notify_cancel(token: i32) -> u32;
    }

    unsafe fn cf_str(s: &str) -> Ref {
        let c = CString::new(s).expect("no NUL in key");
        CFStringCreateWithCString(kCFAllocatorDefault, c.as_ptr(), UTF8)
    }

    unsafe fn cf_i32(v: i32) -> Ref {
        CFNumberCreate(kCFAllocatorDefault, SINT32, &v as *const i32 as *const c_void)
    }

    unsafe fn to_string(s: Ref) -> Option<String> {
        if s.is_null() || CFGetTypeID(s) != CFStringGetTypeID() {
            return None;
        }
        let mut buf = [0 as c_char; 256];
        (CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as isize, UTF8) != 0)
            .then(|| CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
    }

    pub struct Sensors {
        client: Ref,
        services: Ref,
        list: Vec<(String, Ref)>, // services are kept alive by `services`
    }

    impl Sensors {
        pub fn open() -> Option<Self> {
            unsafe {
                let client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
                if client.is_null() {
                    return None;
                }
                // Apple vendor usage page 0xff00, usage 5: temperature sensors.
                let keys = [cf_str("PrimaryUsagePage"), cf_str("PrimaryUsage")];
                let values = [cf_i32(0xff00), cf_i32(5)];
                let matching = CFDictionaryCreate(
                    kCFAllocatorDefault,
                    keys.as_ptr(),
                    values.as_ptr(),
                    2,
                    addr_of!(kCFTypeDictionaryKeyCallBacks),
                    addr_of!(kCFTypeDictionaryValueCallBacks),
                );
                for r in keys.iter().chain(values.iter()) {
                    CFRelease(*r);
                }
                IOHIDEventSystemClientSetMatching(client, matching);
                CFRelease(matching);

                let services = IOHIDEventSystemClientCopyServices(client);
                if services.is_null() {
                    CFRelease(client);
                    return None;
                }
                let product = cf_str("Product");
                let mut list = Vec::new();
                for i in 0..CFArrayGetCount(services) {
                    let service = CFArrayGetValueAtIndex(services, i);
                    let name = IOHIDServiceClientCopyProperty(service, product);
                    if let Some(n) = to_string(name) {
                        list.push((n, service));
                    }
                    if !name.is_null() {
                        CFRelease(name);
                    }
                }
                CFRelease(product);

                if list.is_empty() {
                    CFRelease(services);
                    CFRelease(client);
                    return None;
                }
                Some(Sensors { client, services, list })
            }
        }

        pub fn read(&self) -> Vec<Reading> {
            let field = (EVENT_TEMPERATURE as u32) << 16;
            self.list
                .iter()
                .filter_map(|(name, service)| unsafe {
                    let event = IOHIDServiceClientCopyEvent(*service, EVENT_TEMPERATURE, 0, 0);
                    if event.is_null() {
                        return None;
                    }
                    let t = IOHIDEventGetFloatValue(event, field);
                    CFRelease(event);
                    // Some sensors report 0 or junk; keep plausible values only.
                    (t > 1.0 && t < 150.0).then(|| Reading { name: name.clone(), celsius: t })
                })
                .collect()
        }
    }

    impl Drop for Sensors {
        fn drop(&mut self) {
            unsafe {
                CFRelease(self.services);
                CFRelease(self.client);
            }
        }
    }

    /// 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping.
    pub fn thermal_pressure() -> Option<u64> {
        unsafe {
            let mut token = 0i32;
            if notify_register_check(c"com.apple.system.thermalpressurelevel".as_ptr(), &mut token) != 0 {
                return None;
            }
            let mut state = 0u64;
            let ok = notify_get_state(token, &mut state) == 0;
            notify_cancel(token);
            ok.then_some(state)
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::Reading;

    pub struct Sensors;

    impl Sensors {
        pub fn open() -> Option<Self> {
            None
        }
        pub fn read(&self) -> Vec<Reading> {
            Vec::new()
        }
    }

    pub fn thermal_pressure() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(name: &str, celsius: f64) -> Reading {
        Reading { name: name.into(), celsius }
    }

    #[test]
    fn groups_known_sensor_names() {
        assert_eq!(category("pACC MTR Temp Sensor2"), Some("CPU P-cores"));
        assert_eq!(category("eACC MTR Temp Sensor0"), Some("CPU E-cores"));
        assert_eq!(category("PMU tdie3"), Some("CPU die"));
        assert_eq!(category("GPU MTR Temp Sensor1"), Some("GPU"));
        assert_eq!(category("NAND CH0 temp"), Some(INTERNAL_SSD));
        assert_eq!(category("PMU tcal"), None);
    }

    #[test]
    fn summary_tracks_first_last_peak_and_worst_pressure() {
        let mut s = Summary::default();
        s.update(&[r("NAND CH0 temp", 40.0), r("pACC MTR Temp Sensor0", 50.0), r("pACC MTR Temp Sensor1", 55.0)], Some(0));
        s.update(&[r("NAND CH0 temp", 46.0), r("pACC MTR Temp Sensor0", 88.0)], Some(2));
        s.update(&[r("NAND CH0 temp", 44.0), r("pACC MTR Temp Sensor0", 70.0)], Some(1));

        assert_eq!(s.spans[0].0, "CPU P-cores"); // CATEGORIES order, not insertion order
        let cpu = s.spans[0].1;
        assert_eq!((cpu.first, cpu.last, cpu.peak), (55.0, 70.0, 88.0));
        let ssd = s.spans[1].1;
        assert_eq!((ssd.first, ssd.last, ssd.peak), (40.0, 44.0, 46.0));
        assert_eq!(s.worst_pressure, Some(2));
        assert_eq!(s.peak_where(|c| c.starts_with("CPU")), Some(88.0));
    }
}
