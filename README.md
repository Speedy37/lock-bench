A benchmark of mutex vs spinlock throughput for an extremely short critical section under varying levels of contention on "average" desktop.

Inspiration and code for `AmdSpinlock` are from https://probablydance.com/2019/12/30/measuring-mutexes-spinlocks-and-how-bad-the-linux-scheduler-really-is/.

Summary of results:

* Spinlocks are almost always significantly worse than a good mutex, and never significantly better,
* Contention makes spinlocks relatively slower.

Biggest known caveat (apart from this being a single benchmark run on a single machine):

The best mutex implementation seems to be relatively more optimized than the best spinlock implementation.

## Results

https://speedy37.github.io/lock-bench/