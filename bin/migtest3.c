/*
 * migtest3.c - MattX Batch 1 Syscall QA Tester
 * Tests: getpid, gettid, uname, getrandom, prlimit64, arch_prctl, prctl
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/utsname.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <string.h>

void worker_process() {
    printf("[Worker] Started with PID: %d\n", getpid());
    
    while (1) {
        printf("\n--- Batch 1 Syscall Test ---\n");
        
        // 1. getpid & gettid
        pid_t pid = getpid();
        pid_t tid = syscall(SYS_gettid);
        printf("[Worker] getpid() = %d, gettid() = %d\n", pid, tid);
        
        // 2. uname
        struct utsname uts;
        if (uname(&uts) == 0) {
            printf("[Worker] uname() -> nodename: %s, release: %s\n", uts.nodename, uts.release);
        } else {
            perror("[Worker] uname failed");
        }
        
        // 3. getrandom
        unsigned int rand_val = 0;
        long res = syscall(SYS_getrandom, &rand_val, sizeof(rand_val), 0);
        if (res > 0) {
            printf("[Worker] getrandom() -> 0x%08x\n", rand_val);
        } else {
            perror("[Worker] getrandom failed");
        }
        
        // 4. prlimit64 (using prlimit wrapper)
        struct rlimit rlim;
        if (prlimit(0, RLIMIT_NOFILE, NULL, &rlim) == 0) {
            printf("[Worker] prlimit64(RLIMIT_NOFILE) -> soft: %lu, hard: %lu\n", 
                   (unsigned long)rlim.rlim_cur, (unsigned long)rlim.rlim_max);
        } else {
            perror("[Worker] prlimit64 failed");
        }
        
        // 5. arch_prctl
        unsigned long fs_base = 0;
        if (syscall(SYS_arch_prctl, ARCH_GET_FS, &fs_base) == 0) {
            printf("[Worker] arch_prctl(ARCH_GET_FS) -> 0x%lx\n", fs_base);
        } else {
            perror("[Worker] arch_prctl failed");
        }
        
        // 6. prctl
        int dumpable = prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
        if (dumpable >= 0) {
            printf("[Worker] prctl(PR_GET_DUMPABLE) -> %d\n", dumpable);
        } else {
            perror("[Worker] prctl failed");
        }
        
        printf("----------------------------\n");
        
        // Sleep for 3 seconds to give Matt time to echo into /proc/mattx/admin!
        sleep(3); 
    }
}

int main() {
    printf("[Master] Starting migtest3 (Batch 1 Syscall QA)...\n");
    
    pid_t pid = fork();
    if (pid < 0) {
        perror("Fork failed");
        return 1;
    }
    
    if (pid == 0) {
        // Child process
        worker_process();
        exit(0);
    } else {
        // Parent process
        printf("[Master] Spawned worker with PID %d. Waiting for it to finish...\n", pid);
        wait(NULL);
        printf("[Master] Worker finished. Exiting.\n");
    }
    
    return 0;
}
