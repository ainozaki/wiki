"""
pingの結果からRTTを抜き出してstdoutするスクリプト

$ cat ping_result.txt
PING 8.8.8.8 (8.8.8.8): 56 data bytes
64 bytes from 8.8.8.8: icmp_seq=0 ttl=55 time=8.818 ms
64 bytes from 8.8.8.8: icmp_seq=1 ttl=55 time=4.524 ms
64 bytes from 8.8.8.8: icmp_seq=2 ttl=55 time=18.850 ms
64 bytes from 8.8.8.8: icmp_seq=3 ttl=55 time=6.007 ms

--- 8.8.8.8 ping statistics ---
4 packets transmitted, 4 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 4.524/9.550/18.850/5.587 ms

$ python3 ./ping_extract_rtt.py ./ping_result.txt
8.818
4.524
18.850
6.007
"""

import sys

def print_rtt(line):
    # expected format: "64 bytes from 1.1.1.1: icmp_seq=0 ttl=54 time=4.722 ms"
    splited = line.split()
    if len(splited) < 8:
        return
    if splited[1] != "bytes":
        return
    rtt = splited[6]
    print(rtt[5:])

def main():
    argv = sys.argv
    if len(argv) < 2:
        print("usage:", argv[0], "<filename of ping result>")
        return
    
    # file open
    filepath = argv[1]
    with open(filepath) as f:
        for line in f:
            print_rtt(line.strip())


if __name__ == "__main__":
    main()