//! SSD health from the NVMe "SMART / Health Information" log (log page 0x02):
//! life used, bytes written, spare blocks, power-on hours, error counts.
//!
//! macOS gives apps this data for NVMe drives on PCIe-class links: the internal
//! SSD and Thunderbolt / USB4 enclosures. Over USB 3.x the enclosure's bridge chip
//! sits behind Apple's mass-storage driver, which doesn't pass SMART commands
//! through, so those drives are listed but their health can't be read (smartctl
//! has the same limit). The IOKit calls mirror smartmontools' os_darwin.cpp.

use std::path::Path;
use std::process::Command;

pub struct Drive {
    /// Whole physical disk, e.g. "disk0".
    pub bsd: String,
    /// Other whole disks on the same SSD (Apple Silicon splits it into NVMe namespaces).
    pub aliases: Vec<String>,
    pub model: String,
    /// "Apple Fabric", "PCI-Express", "USB", "Thunderbolt", ...
    pub interconnect: String,
    /// "Internal" or "External".
    pub location: String,
    pub size: u64,
    pub health: Result<Health, String>,
    handle: imp::Handle,
}

impl Drive {
    /// Reads the health log again (e.g. for a fresh temperature).
    pub fn refresh(&self) -> Option<Health> {
        self.handle.read().ok()
    }

    pub fn is_internal(&self) -> bool {
        self.location.eq_ignore_ascii_case("internal")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Health {
    pub critical_warning: u8,
    pub temperature_c: Option<f64>,
    pub available_spare: u8,
    pub spare_threshold: u8,
    /// Controller's estimate of rated endurance used; can exceed 100.
    pub percent_used: u8,
    pub bytes_read: u128,
    pub bytes_written: u128,
    pub power_cycles: u128,
    pub power_on_hours: u128,
    pub unsafe_shutdowns: u128,
    pub media_errors: u128,
    pub error_log_entries: u128,
}

/// NVMe counts data in units of 1000 × 512 bytes.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))] // used by the macOS implementation
const DATA_UNIT: u128 = 512_000;

impl Health {
    /// Parses the 512-byte NVMe SMART / Health Information log.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))] // used by the macOS implementation
    pub fn parse(log: &[u8]) -> Option<Self> {
        if log.len() < 512 {
            return None;
        }
        let u128_at = |off: usize| u128::from_le_bytes(log[off..off + 16].try_into().unwrap());
        let kelvin = u16::from_le_bytes([log[1], log[2]]);
        Some(Health {
            critical_warning: log[0],
            temperature_c: (kelvin > 0).then_some(kelvin as f64 - 273.15),
            available_spare: log[3],
            spare_threshold: log[4],
            percent_used: log[5],
            bytes_read: u128_at(32).saturating_mul(DATA_UNIT),
            bytes_written: u128_at(48).saturating_mul(DATA_UNIT),
            power_cycles: u128_at(112),
            power_on_hours: u128_at(128),
            unsafe_shutdowns: u128_at(144),
            media_errors: u128_at(160),
            error_log_entries: u128_at(176),
        })
    }

    /// Human-readable list of the critical-warning bits that are set.
    pub fn warnings(&self) -> Vec<&'static str> {
        const BITS: [&str; 6] = [
            "spare blocks below threshold",
            "temperature out of range",
            "reliability degraded",
            "drive switched to read-only",
            "volatile memory backup failed",
            "persistent memory region read-only",
        ];
        BITS.iter()
            .enumerate()
            .filter(|(bit, _)| self.critical_warning & (1 << bit) != 0)
            .map(|(_, s)| *s)
            .collect()
    }
}

/// Every physical disk macOS knows about, ordered by disk number.
pub fn list() -> Vec<Drive> {
    imp::list()
}

/// The physical drive behind "disk4", "/dev/rdisk4s1", or any path on a mounted volume.
pub fn find(target: &str) -> Option<Drive> {
    let dev = target.strip_prefix("/dev/").unwrap_or(target);
    let dev = dev.strip_prefix('r').filter(|d| d.starts_with("disk")).unwrap_or(dev);
    let bsd = if dev.starts_with("disk") { dev.to_string() } else { bsd_for_path(Path::new(target))? };
    imp::for_bsd(&bsd)
}

/// "/Volumes/Foo" → "disk4s1", via df (the device of the volume holding the path).
fn bsd_for_path(path: &Path) -> Option<String> {
    let out = Command::new("df").arg("-P").arg(path).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let dev = text.lines().nth(1)?.split_whitespace().next()?;
    dev.strip_prefix("/dev/").map(str::to_string)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))] // used by the macOS implementation
fn disk_number(bsd: &str) -> u32 {
    bsd.trim_start_matches("disk")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or(u32::MAX)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))] // used by the macOS implementation
fn no_smart_reason(interconnect: &str) -> String {
    if interconnect.to_ascii_uppercase().contains("USB") {
        "not readable: macOS doesn't pass health data through USB 3.x enclosures \
         (Thunderbolt and USB4 ones work)"
            .into()
    } else {
        "not readable: this drive doesn't offer NVMe health data to macOS".into()
    }
}

// ---------------------------------------------------------------------------
// macOS implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod imp {
    use super::{disk_number, no_smart_reason, Drive, Health};
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr::null_mut;

    type Ref = *const c_void;
    type Obj = u32; // io_object_t

    const MAIN_PORT: u32 = 0; // kIOMainPortDefault is MACH_PORT_NULL
    const SINT64: isize = 4; // kCFNumberSInt64Type
    const UTF8: u32 = 0x0800_0100;
    const HEALTH_LOG_PAGE: u32 = 0x02;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CFUUIDBytes([u8; 16]);

    // From smartmontools os_darwin.h (Apple's NVMeSMARTLibExternal.h).
    const NVME_SMART_USER_CLIENT_TYPE: CFUUIDBytes = CFUUIDBytes([
        0xAA, 0x0F, 0xA6, 0xF9, 0xC2, 0xD6, 0x45, 0x7F, 0xB1, 0x0B, 0x59, 0xA1, 0x32, 0x53, 0x29, 0x2F,
    ]);
    const NVME_SMART_INTERFACE: CFUUIDBytes = CFUUIDBytes([
        0xCC, 0xD1, 0xDB, 0x19, 0xFD, 0x9A, 0x4D, 0xAF, 0xBF, 0x95, 0x12, 0x45, 0x4B, 0x23, 0x0A, 0xB6,
    ]);
    // kIOCFPlugInInterfaceID: C244E858-109C-11D4-91D4-0050E4C6426F
    const CF_PLUGIN_INTERFACE: CFUUIDBytes = CFUUIDBytes([
        0xC2, 0x44, 0xE8, 0x58, 0x10, 0x9C, 0x11, 0xD4, 0x91, 0xD4, 0x00, 0x50, 0xE4, 0xC6, 0x42, 0x6F,
    ]);

    type QueryInterfaceFn = unsafe extern "C" fn(*mut c_void, CFUUIDBytes, *mut *mut c_void) -> i32;
    type RefCountFn = unsafe extern "C" fn(*mut c_void) -> u32;

    /// IOCFPlugInInterface — only the IUnknown part is used.
    #[repr(C)]
    struct PlugInVtbl {
        _reserved: *mut c_void,
        query_interface: QueryInterfaceFn,
        _add_ref: RefCountFn,
        _release: RefCountFn,
    }

    /// IONVMeSMARTInterface, up to GetLogPage.
    #[repr(C)]
    struct SmartVtbl {
        _reserved: *mut c_void,
        _query_interface: QueryInterfaceFn,
        _add_ref: RefCountFn,
        release: RefCountFn,
        _version: u16,
        _revision: u16,
        smart_read_data: unsafe extern "C" fn(*mut c_void, *mut u8) -> i32,
        _get_identify_data: unsafe extern "C" fn(*mut c_void, *mut u8, u32) -> i32,
        _reserved0: u64,
        _reserved1: u64,
        get_log_page: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32) -> i32,
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFAllocatorDefault: Ref;
        fn CFStringCreateWithCString(alloc: Ref, s: *const c_char, enc: u32) -> Ref;
        fn CFStringGetCString(s: Ref, buf: *mut c_char, size: isize, enc: u32) -> u8;
        fn CFStringGetTypeID() -> usize;
        fn CFBooleanGetTypeID() -> usize;
        fn CFBooleanGetValue(b: Ref) -> u8;
        fn CFNumberGetTypeID() -> usize;
        fn CFNumberGetValue(n: Ref, kind: isize, value: *mut c_void) -> u8;
        fn CFDictionaryGetTypeID() -> usize;
        fn CFDictionaryGetValue(d: Ref, key: Ref) -> Ref;
        fn CFGetTypeID(cf: Ref) -> usize;
        fn CFUUIDCreateFromUUIDBytes(alloc: Ref, bytes: CFUUIDBytes) -> Ref;
        fn CFRelease(cf: Ref);
    }

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOServiceMatching(class: *const c_char) -> Ref;
        fn IOBSDNameMatching(main: u32, options: u32, bsd: *const c_char) -> Ref;
        fn IOServiceGetMatchingService(main: u32, matching: Ref) -> Obj;
        fn IOServiceGetMatchingServices(main: u32, matching: Ref, iter: *mut Obj) -> i32;
        fn IOIteratorNext(iter: Obj) -> Obj;
        fn IOObjectRetain(obj: Obj) -> i32;
        fn IOObjectRelease(obj: Obj) -> i32;
        fn IOObjectConformsTo(obj: Obj, class: *const c_char) -> i32;
        fn IORegistryEntryGetParentEntry(entry: Obj, plane: *const c_char, parent: *mut Obj) -> i32;
        fn IORegistryEntryGetRegistryEntryID(entry: Obj, id: *mut u64) -> i32;
        fn IORegistryEntryCreateCFProperty(entry: Obj, key: Ref, alloc: Ref, options: u32) -> Ref;
        fn IOCreatePlugInInterfaceForService(
            service: Obj,
            plugin_type: Ref,
            interface_type: Ref,
            the_interface: *mut *mut *mut PlugInVtbl,
            score: *mut i32,
        ) -> i32;
        fn IODestroyPlugInInterface(interface: *mut *mut PlugInVtbl) -> i32;
    }

    /// An owned CoreFoundation reference.
    struct Cf(Ref);

    impl Drop for Cf {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) }
            }
        }
    }

    fn cf_str(s: &str) -> Cf {
        let c = CString::new(s).expect("no NUL in key");
        Cf(unsafe { CFStringCreateWithCString(kCFAllocatorDefault, c.as_ptr(), UTF8) })
    }

    unsafe fn to_string(s: Ref) -> Option<String> {
        if s.is_null() || CFGetTypeID(s) != CFStringGetTypeID() {
            return None;
        }
        let mut buf = [0 as c_char; 512];
        (CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as isize, UTF8) != 0)
            .then(|| CStr::from_ptr(buf.as_ptr()).to_string_lossy().trim().to_string())
    }

    /// An owned IOKit registry object.
    struct Object(Obj);

    impl Drop for Object {
        fn drop(&mut self) {
            if self.0 != 0 {
                unsafe { IOObjectRelease(self.0) };
            }
        }
    }

    impl Object {
        fn from_bsd(bsd: &str) -> Option<Object> {
            let name = CString::new(bsd).ok()?;
            // IOServiceGetMatchingService consumes the matching dictionary.
            let obj = unsafe {
                IOServiceGetMatchingService(MAIN_PORT, IOBSDNameMatching(MAIN_PORT, 0, name.as_ptr()))
            };
            (obj != 0).then_some(Object(obj))
        }

        fn retained(&self) -> Object {
            unsafe { IOObjectRetain(self.0) };
            Object(self.0)
        }

        fn parent(&self) -> Option<Object> {
            let mut p: Obj = 0;
            let kr = unsafe { IORegistryEntryGetParentEntry(self.0, c"IOService".as_ptr(), &mut p) };
            (kr == 0 && p != 0).then_some(Object(p))
        }

        fn conforms(&self, class: &CStr) -> bool {
            unsafe { IOObjectConformsTo(self.0, class.as_ptr()) != 0 }
        }

        fn id(&self) -> u64 {
            let mut id = 0u64;
            unsafe { IORegistryEntryGetRegistryEntryID(self.0, &mut id) };
            id
        }

        fn property(&self, key: &str) -> Option<Cf> {
            let k = cf_str(key);
            let v = unsafe { IORegistryEntryCreateCFProperty(self.0, k.0, kCFAllocatorDefault, 0) };
            (!v.is_null()).then_some(Cf(v))
        }

        fn string(&self, key: &str) -> Option<String> {
            unsafe { to_string(self.property(key)?.0) }
        }

        fn number(&self, key: &str) -> Option<i64> {
            let v = self.property(key)?;
            let mut n = 0i64;
            unsafe {
                (CFGetTypeID(v.0) == CFNumberGetTypeID()
                    && CFNumberGetValue(v.0, SINT64, &mut n as *mut i64 as *mut c_void) != 0)
                    .then_some(n)
            }
        }

        fn boolean(&self, key: &str) -> Option<bool> {
            let v = self.property(key)?;
            unsafe { (CFGetTypeID(v.0) == CFBooleanGetTypeID()).then(|| CFBooleanGetValue(v.0) != 0) }
        }

        /// A string inside a dictionary property, e.g. "Device Characteristics" → "Product Name".
        fn dict_string(&self, dict: &str, key: &str) -> Option<String> {
            let d = self.property(dict)?;
            let k = cf_str(key);
            unsafe {
                if CFGetTypeID(d.0) != CFDictionaryGetTypeID() {
                    return None;
                }
                to_string(CFDictionaryGetValue(d.0, k.0)).filter(|s| !s.is_empty())
            }
        }

        fn is_whole_media(&self) -> bool {
            self.conforms(c"IOMedia") && self.boolean("Whole") == Some(true)
        }
    }

    /// The service that offers the NVMe SMART plug-in, if any. It is either the
    /// NVMe block-storage device itself or one of its first few ancestors.
    pub struct Handle(Option<Object>);

    impl Handle {
        pub fn read(&self) -> Result<Health, String> {
            match &self.0 {
                Some(service) => read_health(service),
                None => Err("no health data available".into()),
            }
        }
    }

    fn read_health(service: &Object) -> Result<Health, String> {
        unsafe {
            let plugin_type = Cf(CFUUIDCreateFromUUIDBytes(kCFAllocatorDefault, NVME_SMART_USER_CLIENT_TYPE));
            let interface_type = Cf(CFUUIDCreateFromUUIDBytes(kCFAllocatorDefault, CF_PLUGIN_INTERFACE));
            let mut plugin: *mut *mut PlugInVtbl = null_mut();
            let mut score = 0i32;
            let kr = IOCreatePlugInInterfaceForService(
                service.0,
                plugin_type.0,
                interface_type.0,
                &mut plugin,
                &mut score,
            );
            if kr != 0 || plugin.is_null() {
                return Err(format!("couldn't open the NVMe health interface (IOKit error {kr:#x})"));
            }

            let mut smart: *mut *mut SmartVtbl = null_mut();
            let hr = ((**plugin).query_interface)(
                plugin.cast(),
                NVME_SMART_INTERFACE,
                (&mut smart as *mut *mut *mut SmartVtbl).cast(),
            );
            let result = if hr != 0 || smart.is_null() {
                Err(format!("NVMe health interface unavailable (error {hr:#x})"))
            } else {
                let mut log = [0u8; 512];
                // Same call smartctl uses: log page 0x02, length in dwords, zero-based.
                let mut kr = ((**smart).get_log_page)(
                    smart.cast(),
                    log.as_mut_ptr().cast(),
                    HEALTH_LOG_PAGE,
                    (log.len() / 4 - 1) as u32,
                );
                if kr != 0 {
                    kr = ((**smart).smart_read_data)(smart.cast(), log.as_mut_ptr());
                }
                ((**smart).release)(smart.cast());
                if kr != 0 {
                    Err(format!("reading the health log failed (IOKit error {kr:#x})"))
                } else {
                    Health::parse(&log).ok_or_else(|| "health log was malformed".to_string())
                }
            };
            IODestroyPlugInInterface(plugin);
            result
        }
    }

    /// Walks up from any BSD disk ("disk3s5") to the physical drive and returns
    /// (its whole-disk IOMedia, its IOBlockStorageDevice). APFS volumes sit in
    /// synthesized disks whose ancestors include the physical partition, so the
    /// last whole IOMedia before the block device is the real disk.
    fn physical(bsd: &str) -> Option<(Object, Object)> {
        let mut cur = Object::from_bsd(bsd)?;
        let mut whole: Option<Object> = None;
        loop {
            if cur.conforms(c"IOBlockStorageDevice") {
                return Some((whole?, cur));
            }
            let parent = cur.parent()?;
            if cur.is_whole_media() {
                whole = Some(cur);
            }
            cur = parent;
        }
    }

    fn smart_service(block: &Object) -> Option<Object> {
        let mut cur = block.retained();
        for _ in 0..5 {
            if cur.property("NVMe SMART Capable").is_some() {
                return Some(cur);
            }
            cur = cur.parent()?;
        }
        None
    }

    fn drive(whole: &Object, block: &Object) -> Option<Drive> {
        let bsd = whole.string("BSD Name")?;
        let model = ["Vendor Name", "Product Name"]
            .iter()
            .filter_map(|k| block.dict_string("Device Characteristics", k))
            .collect::<Vec<_>>()
            .join(" ");
        let interconnect = block
            .dict_string("Protocol Characteristics", "Physical Interconnect")
            .unwrap_or_else(|| "unknown".into());
        let location = block
            .dict_string("Protocol Characteristics", "Physical Interconnect Location")
            .unwrap_or_else(|| "unknown".into());

        let handle = Handle(smart_service(block));
        let health = match handle.0 {
            Some(_) => handle.read(),
            None => Err(no_smart_reason(&interconnect)),
        };
        Some(Drive {
            bsd,
            aliases: Vec::new(),
            model: if model.is_empty() { "unknown model".into() } else { model },
            interconnect,
            location,
            size: whole.number("Size").unwrap_or(0).max(0) as u64,
            health,
            handle,
        })
    }

    pub fn for_bsd(bsd: &str) -> Option<Drive> {
        let (whole, block) = physical(bsd)?;
        drive(&whole, &block)
    }

    /// The block-storage device a whole disk sits on, if it is a physical disk.
    /// The tree is device → IOBlockStorageDriver → IOMedia, so look two levels up;
    /// synthesized disks (APFS containers) have a partition scheme there instead.
    fn block_device_of(media: &Object) -> Option<Object> {
        let mut cur = media.parent()?;
        for _ in 0..2 {
            if cur.conforms(c"IOBlockStorageDevice") {
                return Some(cur);
            }
            if cur.conforms(c"IOMedia") {
                return None;
            }
            cur = cur.parent()?;
        }
        None
    }

    pub fn list() -> Vec<Drive> {
        // Physical whole disks: a whole IOMedia whose driver sits on a block-storage device.
        let mut found: Vec<(u32, String, Object, Object)> = Vec::new();
        unsafe {
            let mut iter: Obj = 0;
            if IOServiceGetMatchingServices(MAIN_PORT, IOServiceMatching(c"IOMedia".as_ptr()), &mut iter) != 0 {
                return Vec::new();
            }
            let iter = Object(iter);
            loop {
                let next = IOIteratorNext(iter.0);
                if next == 0 {
                    break;
                }
                let media = Object(next);
                if !media.is_whole_media() {
                    continue;
                }
                let Some(block) = block_device_of(&media) else { continue };
                let virtual_disk = block
                    .dict_string("Protocol Characteristics", "Physical Interconnect")
                    .is_some_and(|i| i.contains("Virtual"));
                if virtual_disk {
                    continue; // mounted disk images
                }
                let Some(bsd) = media.string("BSD Name") else { continue };
                found.push((disk_number(&bsd), bsd, media, block));
            }
        }
        found.sort_by_key(|(n, ..)| *n);

        // Namespaces of one SSD share a controller; show them as one drive.
        let mut drives: Vec<(u64, Drive)> = Vec::new();
        for (_, bsd, media, block) in found {
            let controller = block.parent().map(|p| p.id()).unwrap_or_else(|| block.id());
            if let Some((_, d)) = drives.iter_mut().find(|(c, _)| *c == controller) {
                d.aliases.push(bsd);
            } else if let Some(d) = drive(&media, &block) {
                drives.push((controller, d));
            }
        }
        drives.into_iter().map(|(_, d)| d).collect()
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::{Drive, Health};

    pub struct Handle;

    impl Handle {
        pub fn read(&self) -> Result<Health, String> {
            Err("drive health is read on macOS only".into())
        }
    }

    pub fn for_bsd(_bsd: &str) -> Option<Drive> {
        None
    }

    pub fn list() -> Vec<Drive> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvme_health_log() {
        let mut log = [0u8; 512];
        log[0] = 0b0000_0101; // spare low + reliability degraded
        log[1..3].copy_from_slice(&310u16.to_le_bytes()); // 310 K
        log[3] = 100;
        log[4] = 99;
        log[5] = 3;
        log[32..48].copy_from_slice(&40_000_000u128.to_le_bytes()); // data units read
        log[48..64].copy_from_slice(&25_000_000u128.to_le_bytes()); // data units written
        log[112..128].copy_from_slice(&812u128.to_le_bytes());
        log[128..144].copy_from_slice(&1234u128.to_le_bytes());
        log[144..160].copy_from_slice(&21u128.to_le_bytes());

        let h = Health::parse(&log).unwrap();
        assert_eq!(h.percent_used, 3);
        assert_eq!(h.bytes_written, 25_000_000 * 512_000); // 12.8 TB
        assert_eq!(h.bytes_read, 40_000_000 * 512_000);
        assert_eq!((h.power_cycles, h.power_on_hours, h.unsafe_shutdowns), (812, 1234, 21));
        assert!((h.temperature_c.unwrap() - 36.85).abs() < 1e-9);
        assert_eq!(h.warnings(), vec!["spare blocks below threshold", "reliability degraded"]);
        assert!(Health::parse(&log[..100]).is_none());
    }

    #[test]
    fn orders_disks_numerically() {
        assert!(disk_number("disk2") < disk_number("disk10"));
        assert_eq!(disk_number("disk4s1"), 4);
    }
}
