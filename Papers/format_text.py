import glob
import os
import re

# find all txt files on disk
txt_files = glob.glob('*.txt')

# loop through each text file
# read in lines, split docment by \n\n, write to new file 
for txt_file in txt_files:

    # get four digit from filename
    year = re.findall(r'\d{4}', txt_file)[0]

    # get author name before year
    author = txt_file.split(year)[0]

    author_cite = f"{author} et. al ({year}) "

    with open(txt_file, 'r') as f:
        # read all the txt
        text = f.read()
        # split by \n\n
        paragraphs = text.split('\n\n')
        # for each paragraph replace \n with space
        paragraphs = [p.replace('\n', ' ') for p in paragraphs]

        # for each paragraph replace "we" with author_cite
        paragraphs = [p.replace("we ", author_cite) for p in paragraphs]
        paragraphs = [p.replace("We ", author_cite) for p in paragraphs]

        # replace "our" with author_cite
        paragraphs = [p.replace("our ", author_cite) for p in paragraphs]
        paragraphs = [p.replace("Our ", author_cite) for p in paragraphs]

    # write to new file
    with open(txt_file.replace('.txt', '_clean.txt'), 'w') as f:
        for p in paragraphs:
            f.write(p + '\n\n')