import os
import subprocess

file_content = open('filelist.txt')

file_content = file_content.readlines()
for capture in file_content[0:]:
    print(capture)
    if "img_annotated" not in capture:
        out = subprocess.Popen(['aws', 's3', 'cp', 's3://insights-use1-dev-beltvision-data/' + capture.strip('\n'), 'captures/'],
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)
    
        stdout,stderr = out.communicate()
        print(stdout)


#out = subprocess.Popen(['aws', 's3', 'ls', 's3://insights-use1-dev-beltvision-data//captures', '--recursive', stdout=subprocess.PIPE],
#    stdout=subprocess.PIPE, 
#    stderr=subprocess.STDOUT)

#sort = subprocess.check_output(('sort'), stdin=out.stdout)
#tail = subprocess.check_output(('tail', '-n', '200'), stdin=sort.stdout)

#stdout,stderr = out.communicate()
#print(stdout)
#print(stderr)
#print(tail)
#command = "aws s3 ls s3://insights-use1-dev-beltvision-data/captures --recursive | sort | tail -n 200 | awk '{print $4}'"