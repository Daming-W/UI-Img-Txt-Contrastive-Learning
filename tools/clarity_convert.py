import os
import glob
import csv
import argparse

'''
this file aims to convert clarity dataset's original csv file format
into RICOCaption types, to make one image relates to each captions seperately
'''
def convert(input_csv_path,output_csv_path):

    with open(input_csv_path, 'r') as csvfile_in:
        reader = csv.reader(csvfile_in)
        with open(output_csv_path,'w') as csvfile_out:
            writer = csv.writer(csvfile_out)

            for row in list(reader)[1:]:
                for item in row[1:]:
                    if len(item)>5:
                        writer.writerow([row[0],item])
                

input_csv_path='/root/autodl-nas/Clarity-Data/captions.csv'
output_csv_path='/root/autodl-nas/Clarity-Data/captions_sep.csv'
convert(input_csv_path,output_csv_path)