# macbench

A small, dependency-free CPU, memory and SSD benchmark for Apple Silicon Macs. It also reads temperatures and SSD health.

## Run

Download the latest build and run it:

```sh
curl -fsSL -o macbench https://github.com/OWNER/macbench/releases/latest/download/macbench-macos-arm64
chmod +x macbench
./macbench
```

Or build it yourself (needs Rust and Xcode Command Line Tools):

```sh
cargo build --release
./target/release/macbench
```

## Usage

```sh
./macbench                         # full run, about 2 minutes
./macbench --quick                 # about 30 seconds
./macbench --only cpu,mem          # pick sections: cpu, mem, disk
./macbench --dir /Volumes/USB      # benchmark another drive
./macbench --save m1.tsv           # save results
./macbench compare m1.tsv m2.tsv   # compare two machines
./macbench sensors --watch         # live temperatures
./macbench health                  # SSD life used, data written, errors
```

Run `./macbench --help` for all options.

SSD health can be read for the internal SSD and for Thunderbolt or USB4 NVMe drives, but not for USB 3.x drives.
