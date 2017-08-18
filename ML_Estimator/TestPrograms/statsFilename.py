

import pstats
p = pstats.Stats("<statsFilename>")
p.strip_dirs().sort_stats('cumulative').print_stats(30)