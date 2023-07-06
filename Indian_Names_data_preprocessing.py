import pandas as pd

# Indian-Male-Names.csv and Indian-Female-Names.csv contains raw csv of Indian names.
DATADIR = '/Users/xxxxx'
male_names = pd.read_csv('{}/{}'.format(DATADIR,'Indian-Male-Names.csv'))
# male_names contains data available in Indian-Male-Names.csv which contain only unprocessed male names.

female_names = pd.read_csv('{}/{}'.format(DATADIR,'Indian-Female-Names.csv'))
# female_names contains data available in Indian-Female-Names.csv which contain only unprocessed female names.

namelist = []

for names in female_names['name']:
# processing on names available in data.
    first_name = str(names).strip().split(' ')[0]

    namelist.append((first_name, "F"))

for names in male_names['name']:
# processing on names available in data.

    first_name = str(names).strip().split(' ')[0]

    namelist.append((first_name,"M"))

processed_name_list = []

s = 'abcdefghijklmnopqrstuvwxyz'
# s contains all alphabets a-z.

for (i,k) in namelist:
# processing on names acailable in namelist which contains all male and female names.

    i = i.split('@')[0]

    i = i.split('.')[-1]

    i = i.split('-')[-1]

    i = i.split('(')[0]

    i = i.split('/')[0]

    i = i.split('&')[0]

    i = i.split(',')[0]

    i = i.split('[')[0]

    i = i.strip('`').strip()

    if len(i) > 2:

        for j in i:

            if j in s:
                processed_name_list.append((i,k))
 

unique_names = set(processed_name_list)
# unique_names contains unique names and removes repeated names.

processed_name_list = sorted(list(unique_names))
# processed_name_list contains a list of all names in sorted format.
print(processed_name_list[:10])
#print(min(len(p) for (p,q) in processed_name_list))
#With the above line, we figured that 19 characters is the longest for a name in the processed list
data = pd.DataFrame(processed_name_list, columns=['Name', 'Gender'])
# data contains dataframe of processed_name_list.

data.to_csv('{}/{}'.format(DATADIR,'Indian-Names.csv'))
# final output create Indian_Names.csv


