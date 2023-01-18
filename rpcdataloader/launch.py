#!/usr/bin/env python

import argparse

from rpcdataloader import run_worker


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="run RPC worker, prints hostname:port when ready."
    )
    argparser.add_argument("--host", help="binding address")
    argparser.add_argument("--port", type=int, help="binding port port")
    argparser.add_argument("--timeout", type=int, default=60)
    argparser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="maximum number of concurrently active tasks",
    )

    args = argparser.parse_args()

    run_worker(args.host, args.port, args.timeout, args.parallel)
