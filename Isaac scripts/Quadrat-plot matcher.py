import pandas as pd
import os
from moviepy import VideoFileClip

dates_list = ['2023-08-17', '2023-08-23', '2023-08-24', '2023-08-25', '2023-08-28', '2023-08-29', '2023-08-30', '2023-08-31', '2023-09-01']


with open(r"D:\MLS_alignment\missing_plots.txt", 'r') as file:
    missing = file.readlines()
missing.remove(missing[0])
for i, item in enumerate(missing):
    missing[i] = float(item[:4])

quadrat_date_vid = pd.read_excel(r"\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20\MLS_BCI50ha_20x20_checklist.xlsx",
                     sheet_name='BCI50ha_checklist', usecols='A,D,E,F,G,H,N,W')

quadrat_date_vid = quadrat_date_vid[quadrat_date_vid['quadrat'].isin(missing)]
quadrat_date_vid = quadrat_date_vid[~quadrat_date_vid['field_observations'].str.contains('SCANNING', na=False)]
qdv_grouped = quadrat_date_vid.groupby('date_yyyy_mm_dd')
dates_plots = {}

for date in qdv_grouped.groups:
    if str(date)[:10] in dates_list:
        index = list(qdv_grouped.groups[date])
        dates_plots[str(date)[:10]] = list(quadrat_date_vid.loc[index, 'quadrat'])

plots_time = {}
plots_videos = {}
for date in dates_plots:
    for plot in dates_plots[date]:
        plot = str(int(plot))
        while len(plot) != 4:
            plot = '0' + plot
        plots_videos[plot] = list(quadrat_date_vid[quadrat_date_vid['quadrat'] == float(plot)]['video_files'])[0]
        plots_time[plot] = list(quadrat_date_vid[quadrat_date_vid['quadrat'] == float(plot)]['field_observations'])[0]

for date in dates_plots:
    temp = []
    for plot in dates_plots[date]:
        plot = str(int(plot))
        while len(plot) != 4:
            plot = '0' + plot
        temp.append(plot)
    dates_plots[date] = temp

fname_duration = pd.read_excel(r"\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20\MLS_BCI50ha_20x20_checklist.xlsx",
                   sheet_name='BCI50ha_scans_to_check', usecols='A,D')
fname_duration = fname_duration[:127]

files_len = {}
for file in fname_duration['file']:
    files_len[file] = list(fname_duration[fname_duration['file'] == file]['time_duration_seconds'])[0]

dates_files = {}
for file in files_len:
    date = file[:10]
    if date == '2023-07-31':
        date = '2023-08-17'
    elif date == '2023-08-22':
        date = '2023-08-23'
    if date not in dates_files:
        dates_files[date] = []
    dates_files[date].append(file)

path = r'\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20_not_organized_data\videos'
dates_videos = {}
videos_length = {}
videos_start = {}
dirs = os.listdir(path)
dirs.sort()
for folder in dirs:
    if folder == 'Zarchive':
        continue
    date = '20' + folder[:2] + '-' + folder[2:4] + '-' + folder[4:6]
    dates_videos[date] = []
    for file in os.listdir(os.path.join(path, folder)):
        if file.endswith('.MP4'):
            dates_videos[date].append(file)
            clip = VideoFileClip(os.path.join(path, folder, file))
            videos_length[file] = clip.duration

            if len(file) == 17:
                videos_start[file] = file[5:7] + ':' + file[7:9] + ':' + file[9:11]
                plot = file[:4]
                if plot in plots_videos:
                    if not isinstance(plots_videos[plot], str):
                        plots_videos[plot] = file
                    elif file[5:13] not in plots_videos[plot]:
                        plots_videos[plot] = plots_videos[plot] + ';' + file[5:13]
            else:
                videos_start[file] = file[:2] + ':' + file[2:4] + ':' + file[4:6]


print(videos_start)
print(len(videos_start))
print(dates_videos.keys())





# Thoughts organizer
# My goal is to match the files to their quadrats
# To do that there are two steps:
#    1. I must match the files to their videos
#    2. I must match the videos to the quadrats
#
# In order to match the files to their videos, I will use a combination of filtering by the date and comparing scan time to video length
# In order to match the videos to their quadrats, I will use the order in which the videos were created.
#
#
# Data I have: Which plots were done on which days, which videos correspond to some of the plots
# Data I want: Scan duration for each scan, length of each video, start time for each plot,


# CONNECTION I NEED: Plot - File
# To get that, I need: Plot - video, video - file
# To get Plot - Video, I need:
#   Date - plot, Date - video, time - plot, time - video
# To get video - file, I need:
#   Date - file, Date - video, file - length, video - length
# I HAVE:
# Date - plot, Date - file, file - length, plot - start
# I AM MISSING:
# Date - video, video - start, video - length

