# ANSI terminal colour codes -- the single source of truth for every
# coloured print across the project. Any statement using e.g. GREEN can be
# read without decoding a raw escape sequence; add new colours here, not
# inline.

RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'


class Log:
	@staticmethod
	def info(msg):
		print(f'{BOLD}{GREEN}[INFO]{RESET} {msg}')

	@staticmethod
	def warn(msg):
		print(f'{BOLD}{YELLOW}[WARN]{RESET} {msg}')

	@staticmethod
	def section(title):
		# a clear boundary between pipeline phases (e.g. right before a
		# report starts) so scrolling [INFO]/tqdm output doesn't run
		# straight into it
		bar = '=' * 70
		print(f'\n{BOLD}{CYAN}{bar}\n  {title}\n{bar}{RESET}\n')
