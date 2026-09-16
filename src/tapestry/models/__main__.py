"""Run finalized B0 commands, or B1 vintage/masking commands with `b1`."""
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'b1':
    from .b1_run import main
    main(sys.argv[2:])
else:
    from .run import main
    main()
