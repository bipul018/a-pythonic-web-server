# Just a file that chooses which video reader/writer to use

#from .video_write import VideoFromFrameAV as VideoFromFrame
from .video_write import VideoFromFrameCV as VideoFromFrame

from .video_read import FrameGenStreamAV as FrameGenStream
#from .video_read import FrameGenStreamCV as FrameGenStream

