/*
 * migtest3.c - MattX Batch 1 & Batch 2.2 Syscall QA Tester
 * Tests: getpid, gettid, uname, getrandom, prlimit64, arch_prctl, prctl
 *        statfs, fstatfs, newfstatat, faccessat2, readlink, readlinkat
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
#include <sys/vfs.h>    // For statfs
#include <sys/stat.h>   // For stat
#include <fcntl.h>      // For AT_FDCWD and open

// Fallback for faccessat2 if headers are too old (439 on x86_64)
#ifndef SYS_faccessat2
#define SYS_faccessat2 439
#endif

void setup_test_files() {
    printf("[Master] Setting up test files in /tmp...\n");
    system("touch /tmp/mattx_test_file.txt");
    system("ln -sf /tmp/mattx_test_file.txt /tmp/mattx_test_link");
}

void worker_process() {
    printf("[Worker] Started with PID: %d\n", getpid());
    
    while (1) {
        printf("\n--- MattX Syscall QA Test ---\n");
        
        // ==========================================
        // BATCH 1: Identity & Info
        // ==========================================
        pid_t pid = getpid();
        pid_t tid = syscall(SYS_gettid);
        printf("[Worker] getpid() = %d, gettid() = %d\n", pid, tid);
        
        struct utsname uts;
        if (uname(&uts) == 0) {
            printf("[Worker] uname() -> nodename: %s\n", uts.nodename);
        } else { perror("[Worker] uname failed"); }
        
        unsigned int rand_val = 0;
        if (syscall(SYS_getrandom, &rand_val, sizeof(rand_val), 0) > 0) {
            printf("[Worker] getrandom() -> 0x%08x\n", rand_val);
        } else { perror("[Worker] getrandom failed"); }
        
        struct rlimit rlim;
        if (prlimit(0, RLIMIT_NOFILE, NULL, &rlim) == 0) {
            printf("[Worker] prlimit64(RLIMIT_NOFILE) -> soft: %lu\n", (unsigned long)rlim.rlim_cur);
        } else { perror("[Worker] prlimit64 failed"); }
        
        unsigned long fs_base = 0;
        if (syscall(SYS_arch_prctl, ARCH_GET_FS, &fs_base) == 0) {
            printf("[Worker] arch_prctl(ARCH_GET_FS) -> 0x%lx\n", fs_base);
        } else { perror("[Worker] arch_prctl failed"); }
        
        int dumpable = prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
        if (dumpable >= 0) {
            printf("[Worker] prctl(PR_GET_DUMPABLE) -> %d\n", dumpable);
        } else { perror("[Worker] prctl failed"); }

        // ==========================================
        // BATCH 2.2: Metadata Fetchers
        // ==========================================
        
        // 1. statfs
        struct statfs sfs;
        if (statfs("/tmp", &sfs) == 0) {
            printf("[Worker] statfs('/tmp') -> type: 0x%lx\n", (unsigned long)sfs.f_type);
        } else { perror("[Worker] statfs failed"); }

        // 2. fstatfs
        int fd_dir = open("/tmp", O_RDONLY | O_DIRECTORY);
        if (fd_dir >= 0) {
            struct statfs fsfs;
            if (fstatfs(fd_dir, &fsfs) == 0) {
                printf("[Worker] fstatfs(fd) -> type: 0x%lx\n", (unsigned long)fsfs.f_type);
            } else { perror("[Worker] fstatfs failed"); }
            close(fd_dir);
        } else { perror("[Worker] open('/tmp') failed for fstatfs"); }

        // 3. newfstatat (glibc wraps this as fstatat)
        struct stat st;
        if (fstatat(AT_FDCWD, "/tmp/mattx_test_file.txt", &st, 0) == 0) {
            printf("[Worker] newfstatat('/tmp/mattx_test_file.txt') -> size: %lu, inode: %lu\n", 
                   (unsigned long)st.st_size, (unsigned long)st.st_ino);
        } else { perror("[Worker] newfstatat failed"); }

        // 4. faccessat2 (Using raw syscall to bypass glibc fallbacks)
        long acc_res = syscall(SYS_faccessat2, AT_FDCWD, "/tmp/mattx_test_file.txt", R_OK | W_OK, 0);
        if (acc_res == 0) {
            printf("[Worker] faccessat2('/tmp/mattx_test_file.txt') -> Access GRANTED\n");
        } else { perror("[Worker] faccessat2 failed"); }

        // 5. readlink
        char rl_buf[256];
        ssize_t rl_len = readlink("/tmp/mattx_test_link", rl_buf, sizeof(rl_buf) - 1);
        if (rl_len >= 0) {
            rl_buf[rl_len] = '\0';
            printf("[Worker] readlink('/tmp/mattx_test_link') -> '%s'\n", rl_buf);
        } else { perror("[Worker] readlink failed"); }

        // 6. readlinkat
        char rla_buf[256];
        ssize_t rla_len = readlinkat(AT_FDCWD, "/tmp/mattx_test_link", rla_buf, sizeof(rla_buf) - 1);
        if (rla_len >= 0) {
            rla_buf[rla_len] = '\0';
            printf("[Worker] readlinkat(AT_FDCWD, '/tmp/mattx_test_link') -> '%s'\n", rla_buf);
        } else { perror("[Worker] readlinkat failed"); }

        printf("----------------------------\n");
        
        // Sleep for 3 seconds to give Matt time to echo into /proc/mattx/admin!
        sleep(3); 
    }
}

int main() {
    setup_test_files();
    
    printf("[Master] Starting migtest3 (Batch 1 & 2.2 Syscall QA)...\n");
    
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
