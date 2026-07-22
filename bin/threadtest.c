/*
 * threadtest.c - MattX Batch 4 QA Tester (Gang Migration)
 * Tests: clone3, futex, set_tid_address, exit
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/syscall.h>

// The function that our new threads will execute!
void *thread_function(void *arg) {
    int thread_id = *(int *)arg;
    pid_t tid = syscall(SYS_gettid);
    
    printf("  [Thread %d] Born and ticking! (TID: %d)\n", thread_id, tid);
    
    // The thread just stays alive, sleeping until the Master cancels it.
    // (sleep() internally uses futexes and signals!)
    while (1) {
        sleep(1);
    }
    return NULL;
}

void worker_process(int num_threads) {
    printf("[Worker] Started with Mother PID: %d. Target threads: %d\n", getpid(), num_threads);
    
    printf("[Worker] Sleeping 10 seconds for migration prep...\n");
    sleep(10);

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    int *thread_ids = malloc(num_threads * sizeof(int));

    while (1) {
        printf("\n[Worker] --- SPAWNING %d THREADS (clone3 / CLONE_VM) ---\n", num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_ids[i] = i;
            // pthread_create calls clone3() under the hood!
            if (pthread_create(&threads[i], NULL, thread_function, &thread_ids[i]) != 0) {
                perror("Failed to create thread");
            }
        }

        printf("[Worker] Threads spawned. Ticking for 60 seconds...\n");
        for (int i = 0; i < 6; i++) {
            sleep(10);
            printf("[Worker] Tick %d/60 seconds... (Mother PID: %d)\n", (i + 1) * 10, getpid());
        }

        printf("[Worker] --- TEARING DOWN THREADS (futex / exit) ---\n");
        for (int i = 0; i < num_threads; i++) {
            pthread_cancel(threads[i]);
            pthread_join(threads[i], NULL); // This uses futexes to wait for the thread to die!
        }
        
        printf("[Worker] Threads destroyed. Sleeping 10 seconds before next wave...\n");
        sleep(10);
    }
    
    free(threads);
    free(thread_ids);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number_of_threads>\n", argv[0]);
        return 1;
    }
    
    int num_threads = atoi(argv[1]);

    printf("[Master] Starting threadtest (Batch 4 QA)...\n");
    
    pid_t pid = fork();
    if (pid < 0) {
        perror("Fork failed");
        return 1;
    }
    
    if (pid == 0) {
        // Child process
        worker_process(num_threads);
        exit(0);
    } else {
        // Parent process
        printf("[Master] Spawned worker with PID %d. Waiting for it to finish...\n", pid);
        wait(NULL);
        printf("[Master] Worker finished. Exiting.\n");
    }
    
    return 0;
}
