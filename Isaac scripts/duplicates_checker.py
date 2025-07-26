import win32com.client as com
import time
import os

fl_files = []
for folder in os.listdir(r'\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20_not_organized_data\scans'):
    if folder != '~$MLS_BCI50ha_20x20_checklist.xlsx' and folder != 'MLS_BCI50ha_20x20_checklist.xlsx':
        print(folder)
        for file in os.listdir(os.path.join(r'\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20_not_organized_data\scans', folder)):
            if file != 'Zarchive' and not file.endswith('.MP4'):
                fl_files.append(file)
for folder in os.listdir(r'\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20'):
    if folder != '~$MLS_BCI50ha_20x20_checklist.xlsx' and folder != 'MLS_BCI50ha_20x20_checklist.xlsx':
        print(folder)
        for file in os.listdir(os.path.join(r'\\Stri-sm01\ForestLandscapes\LandscapeRaw\MLS\2023\BCI50ha_20x20', folder)):
            if file != 'Zarchive' and not file.endswith('.MP4'):
                fl_files.append(file)

duplicates = []
unique_to_LaCie = []
for file in os.listdir(r'\\Stri-sm01\ForestLandscapes\LaCie\Lab Backup\MLS\data_raw\pre_scans'):
    if file in fl_files:
        duplicates.append(file)
        fl_files.remove(file)
    else:
        unique_to_LaCie.append(file)
print(duplicates)
print(unique_to_LaCie)
print(len(unique_to_LaCie))
print(fl_files)


new_duplicates = [str(len(unique_to_LaCie)) + '\n']
for item in unique_to_LaCie:
    new_duplicates.append(item + '\n')

dup_file = open(os.path.join('D:\MLS_alignment', 'raw_LaCie_unique.txt'), 'w')
dup_file.writelines(new_duplicates)
dup_file.close()



def get_file_path(plot_num, path):
    for filename in os.listdir(path):
        if filename.startswith(plot_num):
            return filename
    print(plot_num, 'FILE NOT FOUND')
    return 'FILE NOT FOUND'

MB = 1024 * 1024
fso = com.Dispatch("Scripting.FileSystemObject")

LaCie_path = r'\\Stri-sm01\ForestLandscapes\LaCie\Lab Backup\MLS\data_processed'
fl_path = r'\\Stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'
t1 = time.time()
fl_unique = []
for folder in os.listdir(fl_path):
    try:
        plot = folder[:4]
        #stats = fso.GetFolder(os.path.join(fl_path, folder))
        if plot in fl_unique:
            print(f'{plot} duplicated in forest_landscapes')
            plot = folder
            fl_unique.append(plot)
            #fl_dict[plot] = stats.Size / MB
        else:
            #fl_dict[plot] = stats.Size / MB
            fl_unique.append(plot)
    except Exception as e:
        print(f'{plot} failed')
        print(e)
t2 = time.time()
LaCie_unique = []
for folder in os.listdir(LaCie_path):
    try:
        plot = folder[:4]
        if plot in LaCie_unique:
            print(f'{plot} duplicated in LaCie')
            plot = folder
            LaCie_unique.append(plot)
        else:
            LaCie_unique.append(plot)
    except Exception as e:
        print(f'{plot} failed')
        print(e)
t3 = time.time()
duplicates = []
exact_duplicates = []
for plot in fl_unique:
    if plot in LaCie_unique:
        duplicates.append(plot)

new_duplicates = [str(len(duplicates)) + '\n']
for item in duplicates:
    new_duplicates.append(item + '\n')

new_LaCie_unique = []
for item in LaCie_unique:
    if item not in duplicates and item not in exact_duplicates:
        new_LaCie_unique.append(item)

new_fl_unique = []
for item in fl_unique:
    if item not in duplicates and item not in exact_duplicates:
        new_fl_unique.append(item)

fl_unique = [str(len(new_fl_unique)) + '\n']
for item in new_fl_unique:
    fl_unique.append(item + '\n')

LaCie_unique = [str(len(new_LaCie_unique)) + '\n']
for item in new_LaCie_unique:
    LaCie_unique.append(item + '\n')

#dup_file = open(os.path.join('D:\MLS_alignment', 'duplicates.txt'), 'w')
#dup_file.writelines(new_duplicates)
#dup_file.close()

#LaCie_unique_file = open(os.path.join('D:\MLS_alignment', 'LaCie_unique.txt'), 'w')
#LaCie_unique_file.writelines(LaCie_unique)
#LaCie_unique_file.close()

#fl_unique_file = open(os.path.join('D:\MLS_alignment', 'fl_unique.txt'), 'w')
#fl_unique_file.writelines(fl_unique)
#fl_unique_file.close()

exact_duplicates = []
for plot in duplicates:
    break
    t1 = time.time()
    folder = get_file_path(plot, LaCie_path)
    LaCie_stat = fso.GetFolder(os.path.join(LaCie_path, folder)).Size / MB
    folder = get_file_path(plot, fl_path)
    fl_stat = fso.GetFolder(os.path.join(fl_path, folder)).Size / MB
    if LaCie_stat == fl_stat:
        exact_duplicates.append(plot)
    else:
        print('====================================================================', plot)
    t2 = time.time()
    print(plot, LaCie_stat, fl_stat, t2 - t1)

#new_exact_duplicates = [str(len(exact_duplicates)) + '\n']
#for item in exact_duplicates:
#    new_exact_duplicates.append(item + '\n')

#ex_dup_file = open(os.path.join('D:\MLS_alignment', 'exact_duplicates.txt'), 'w')
#ex_dup_file.writelines(new_exact_duplicates)
#ex_dup_file.close()

