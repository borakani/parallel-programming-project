#ifndef SCHE_EXECPOSIX_H
#define SCHE_EXECPOSIX_H

#if !defined(_WIN32)

/* argv NULL sonlu dizisi; argv[0] tam yola sahip ikili */
int sche_execposix(const char *cwd, char *const argv[],
                   const char *stdout_path,
                   const char *stderr_path, int *out_status);

#endif /* !_WIN32 */

#endif
