# count up to the given parameter, N
# always print 1 and N
# if
#   N <= 32 print all numbers
#   N > 32, always print even numbers (besides 1 and N)

BEGIN {
    N = ARGV[1];
    if (N <= 32) {
        for (i = 1; i <= N; i++) {
            print i;
        }
    } else {
        step = int(N / 32);
        print 1;
        for (i = 2; i < N; i += 2) {
            if (i % step == 0) {
                print i;
            }
        }
        print N;
    }
}
