"""Four explicit sparse cross-ID fields with train-frequency-20 backoff."""
from final_core import parser, run
if __name__ == "__main__": run(parser(__doc__, "cross_ids").parse_args())
