'''
resize video data to 96x96
require ffmpeg
need to set [Walk, Run, ...] directories under raw_data/
    downloaded from http://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html
'''
import os
import glob

current_path = os.path.dirname(__file__)

resized_path = os.path.join('/media/tetam/One Touch/2232 experiments/alternativeMocogan/mocogan', 'resized_data/')
print(resized_path)
p = "/media/tetam/One Touch/2232 experiments/Grouped By Distance/40/*"

files = glob.glob(os.path.normpath(p))



''' script for cropping '''
for i, file in enumerate(files):
    os.system("ffmpeg -i '%s' -pix_fmt yuv420p -ss 00:00:20 -t 00:00:45 -vf format=gray,crop=512:512:0:219,scale=96:96,eq=contrast=1.5,fps=10 '%s.mp4'" %
            (file, os.path.join(resized_path, str(i))))


#os.system
''' script for reducing size '''
# # resize to 96x76
# for i, file in enumerate(files):
#     os.system("ffmpeg -i %s -pix_fmt yuv420p -vf scale=96:-2 %s.mp4" %
#              (file, os.path.join(resized_path, str(i))))

# files = glob.glob(resized_path+'/*')
# for i, file in enumerate(files):
#     os.system("ffmpeg -y -i %s -pix_fmt yuv420p -vf pad=96:96:0:5 %s" %
#              (file, file))
