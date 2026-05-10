#ifndef _WIN32

#define _POSIX_C_SOURCE 200809L

#include "sche_execposix.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int sche_execposix(const char *cwd, char *const argv[],
                   const char *stdout_path,
                   const char *stderr_path, int *out_status) {
    pid_t pid;
    int st = -1;

    if (!cwd || !argv || !argv[0] || !stdout_path || !stderr_path)
        return -1;

    pid = fork();
    if (pid < 0)
        return -1;

    if (pid == 0) {
        int fd_out, fd_err;
        if (chdir(cwd) != 0) {
            perror("chdir");
            _exit(126);
        }
        fd_out = open(stdout_path,
                      O_CREAT | O_WRONLY | O_TRUNC,
                      0644);
        if (fd_out < 0)
            perror("stdout open");

        fd_err = open(stderr_path,
                      O_CREAT | O_WRONLY | O_TRUNC,
                      0644);
        if (fd_err < 0)
            perror("stderr open");

        if (fd_out >= 0)
            dup2(fd_out, STDOUT_FILENO);
        if (fd_err >= 0)
            dup2(fd_err, STDERR_FILENO);

        if (fd_out >= 0)
            close(fd_out);
        if (fd_err >= 0)
            close(fd_err);

        execv(argv[0], argv);
        perror("execv");
        _exit(127);
    }

    if (waitpid(pid, &st, 0) < 0)
        return -1;
    if (out_status)
        *out_status = WIFEXITED(st) ? WEXITSTATUS(st) : -1;

    return 0;
}

#endif
