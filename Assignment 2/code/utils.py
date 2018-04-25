import sys
import os, errno
import logging

def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        logger.info("  " + key + ": " + str(items[key]))
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.close()
    del args.command