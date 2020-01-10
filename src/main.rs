use std::{iter, sync::Barrier, time};

use criterion::*;
use crossbeam_utils::{thread::scope, CachePadded};

fn mutex(c: &mut Criterion) {
    mutex_timings(c, "duration", TimeKind::Real);
    mutex_timings(c, "cpu", TimeKind::KernelPlusUser);
    mutex_timings(c, "kernel", TimeKind::Kernel);
    mutex_timings(c, "user", TimeKind::User);
}

fn mutex_timings(c: &mut Criterion, name: &str, kind: TimeKind) {
    let mut group = c.benchmark_group(name);
    for input in vec![
        (
            Options {
                n_threads: 32,
                n_locks: 2,
                n_ops: 10000,
            },
            "extreme contention",
        ),
        (
            Options {
                n_threads: 32,
                n_locks: 64,
                n_ops: 10000,
            },
            "heavy contention",
        ),
        (
            Options {
                n_threads: 32,
                n_locks: 1000,
                n_ops: 10000,
            },
            "light contention",
        ),
        (
            Options {
                n_threads: 32,
                n_locks: 1000000,
                n_ops: 10000,
            },
            "no contention",
        ),
    ]
    .iter()
    {
        let (input, p) = input;
        group.bench_with_input(
            BenchmarkId::new(*p, mutexes::Std::LABEL),
            &input,
            |b, options| b.iter_custom(|n| run_bench::<mutexes::Std>(options, n, kind)),
        );
        group.bench_with_input(
            BenchmarkId::new(*p, mutexes::ParkingLot::LABEL),
            &input,
            |b, options| b.iter_custom(|n| run_bench::<mutexes::ParkingLot>(options, n, kind)),
        );
        group.bench_with_input(
            BenchmarkId::new(*p, mutexes::Spin::LABEL),
            &input,
            |b, options| b.iter_custom(|n| run_bench::<mutexes::Spin>(options, n, kind)),
        );
        group.bench_with_input(
            BenchmarkId::new(*p, mutexes::AmdSpin::LABEL),
            &input,
            |b, options| b.iter_custom(|n| run_bench::<mutexes::AmdSpin>(options, n, kind)),
        );
    }
}

#[derive(Debug)]
struct Options {
    n_threads: u32,
    n_locks: u32,
    n_ops: u32,
}

fn random_numbers(seed: u32) -> impl Iterator<Item = u32> {
    let mut random = seed;
    iter::repeat_with(move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    })
}

trait Mutex: Sync + Send + Default {
    const LABEL: &'static str;
    fn with_lock(&self, f: impl FnOnce(&mut u32));
}

#[derive(Debug, Clone, Copy)]
enum TimeKind {
    Kernel,
    User,
    KernelPlusUser,
    Real,
}

struct Times {
    kernel_time: time::Duration,
    user_time: time::Duration,
    elapsed: time::Duration,
}
impl Times {
    fn elapsed(t0: time::Instant, r0: RUsage) -> Times {
        let elapsed = t0.elapsed();
        let r1 = rusage::getrusage();
        Times {
            kernel_time: if r1.kernel_time > r1.kernel_time {
                r1.kernel_time - r0.kernel_time
            } else {
                time::Duration(0, 0)
            },
            user_time: if r1.user_time > r1.user_time {
                r1.user_time - r0.user_time
            } else {
                time::Duration(0, 0)
            },
            elapsed,
        }
    }

    fn get(&self, kind: TimeKind) -> time::Duration {
        match kind {
            TimeKind::Kernel => self.kernel_time.max(time::Duration::new(0, 1)),
            TimeKind::User => self.user_time.max(time::Duration::new(0, 1)),
            TimeKind::KernelPlusUser => {
                (self.kernel_time + self.user_time).max(time::Duration::new(0, 1))
            }
            TimeKind::Real => self.elapsed,
        }
    }
}

fn run_bench<M: Mutex>(options: &Options, n: u64, kind: TimeKind) -> time::Duration {
    let locks = &(0..options.n_locks)
        .map(|_| CachePadded::new(M::default()))
        .collect::<Vec<_>>();

    let start_barrier = &Barrier::new(options.n_threads as usize + 1);
    let end_barrier = &Barrier::new(options.n_threads as usize + 1);

    let elapsed = scope(|scope| {
        let thread_seeds = random_numbers(0x6F4A955E).scan(0x9BA2BF27, |state, n| {
            *state ^= n;
            Some(*state)
        });
        for thread_seed in thread_seeds.take(options.n_threads as usize) {
            scope.spawn(move |_| {
                start_barrier.wait();
                let indexes = random_numbers(thread_seed)
                    .map(|it| it % options.n_locks)
                    .map(|it| it as usize)
                    .take(n as usize);
                for idx in indexes {
                    locks[idx].with_lock(|cnt| *cnt += 1);
                }
                end_barrier.wait();
            });
        }

        std::thread::sleep(time::Duration::from_millis(100));
        start_barrier.wait();
        let r0 = rusage::getrusage();
        let t0 = time::Instant::now();
        end_barrier.wait();
        let elapsed = Times::elapsed(t0, r0);

        let mut total = 0;
        for lock in locks.iter() {
            lock.with_lock(|cnt| total += *cnt);
        }
        assert_eq!(total, options.n_threads * (n as u32));

        elapsed
    })
    .unwrap();
    elapsed.get(kind)
}

#[derive(Debug)]
struct RUsage {
    kernel_time: time::Duration,
    user_time: time::Duration,
}
#[cfg(unix)]
mod rusage {
    use super::RUsage;
    use std::time::Duration;

    fn to_duration(tv: libc::timeval) -> Duration {
        // resolution: 1000ns
        let usec = ((tv.tv_sec as u64) * 1_000_000) + (tv.tv_usec as u64);
        return Duration::new(usec / 1_000_000, ((usec % 1_000_000) as u32) * 1_000);
    }

    pub(crate) fn getrusage() -> RUsage {
        let mut rusage = libc::rusage {
            ru_utime: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_stime: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_maxrss: 0,
            ru_ixrss: 0,
            ru_idrss: 0,
            ru_isrss: 0,
            ru_minflt: 0,
            ru_majflt: 0,
            ru_nswap: 0,
            ru_inblock: 0,
            ru_oublock: 0,
            ru_msgsnd: 0,
            ru_msgrcv: 0,
            ru_nsignals: 0,
            ru_nvcsw: 0,
            ru_nivcsw: 0,
        };
        let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut rusage) };
        if ok != 0 {
            panic!("{:?}", std::io::Error::last_os_error());
        }
        RUsage {
            kernel_time: to_duration(rusage.ru_stime),
            user_time: to_duration(rusage.ru_utime),
        }
    }
}
#[cfg(windows)]
mod rusage {
    use super::RUsage;
    use std::time::Duration;

    use winapi::shared::minwindef::FILETIME;
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::processthreadsapi::GetProcessTimes;

    fn zero() -> FILETIME {
        FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        }
    }

    fn to_duration(ft: FILETIME) -> Duration {
        // resolution: 100ns
        let ft100 = ((ft.dwHighDateTime as u64) << 32) + ft.dwLowDateTime as u64;
        return Duration::new(ft100 / 10_000_000, ((ft100 * 100) % 1000_000_000) as u32);
    }

    pub(crate) fn getrusage() -> RUsage {
        let mut kernel_time = zero();
        let mut user_time = zero();
        let process = unsafe { GetCurrentProcess() };
        let ok = unsafe {
            GetProcessTimes(
                process,
                &mut zero(),
                &mut zero(),
                &mut kernel_time,
                &mut user_time,
            )
        };
        if ok == 0 {
            panic!("{:?}", std::io::Error::last_os_error());
        }
        RUsage {
            kernel_time: to_duration(kernel_time),
            user_time: to_duration(user_time),
        }
    }
}

mod mutexes {
    use super::Mutex;

    pub(crate) type Std = std::sync::Mutex<u32>;
    impl Mutex for Std {
        const LABEL: &'static str = "std::sync::Mutex";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock().unwrap();
            f(&mut guard)
        }
    }

    pub(crate) type ParkingLot = parking_lot::Mutex<u32>;
    impl Mutex for ParkingLot {
        const LABEL: &'static str = "parking_lot::Mutex";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }

    pub(crate) type Spin = spin::Mutex<u32>;
    impl Mutex for Spin {
        const LABEL: &'static str = "spin::Mutex";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }

    pub(crate) type AmdSpin = crate::amd_spinlock::AmdSpinlock<u32>;
    impl Mutex for AmdSpin {
        const LABEL: &'static str = "AmdSpinlock";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }
}

mod amd_spinlock {
    use std::{
        cell::UnsafeCell,
        ops,
        sync::atomic::{spin_loop_hint, AtomicBool, Ordering},
    };

    #[derive(Default)]
    pub(crate) struct AmdSpinlock<T> {
        locked: AtomicBool,
        data: UnsafeCell<T>,
    }
    unsafe impl<T: Send> Send for AmdSpinlock<T> {}
    unsafe impl<T: Send> Sync for AmdSpinlock<T> {}

    pub(crate) struct AmdSpinlockGuard<'a, T> {
        lock: &'a AmdSpinlock<T>,
    }

    impl<T> AmdSpinlock<T> {
        pub(crate) fn lock(&self) -> AmdSpinlockGuard<T> {
            loop {
                let was_locked = self.locked.load(Ordering::Relaxed);
                if !was_locked
                    && self
                        .locked
                        .compare_exchange_weak(
                            was_locked,
                            true,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    break;
                }
                spin_loop_hint()
            }
            AmdSpinlockGuard { lock: self }
        }
    }

    impl<'a, T> ops::Deref for AmdSpinlockGuard<'a, T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            let ptr = self.lock.data.get();
            unsafe { &*ptr }
        }
    }

    impl<'a, T> ops::DerefMut for AmdSpinlockGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            let ptr = self.lock.data.get();
            unsafe { &mut *ptr }
        }
    }

    impl<'a, T> Drop for AmdSpinlockGuard<'a, T> {
        fn drop(&mut self) {
            self.lock.locked.store(false, Ordering::Release)
        }
    }
}

criterion_group!(benches, mutex);
criterion_main!(benches);
