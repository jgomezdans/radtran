#!/usr/bin/python

''' A file that will read the text in the php file and add each array as 
a dictionary item to a dictionary and saves it as a dictionary.
'''
import pickle
import pdb

#file_name = 'lgvalues-abscissa.php'
file_name = 'lgvalues-weights.php'
#text = 'legendre_roots['
text = 'quadrature_weights['
precision = 20

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

mydict = dict()
lst = []
ft = open(file_name,'rt')
file_lines = ft.readlines()
for i, l in enumerate(file_lines):
  if l.find(text) != -1:
    #print l.split()[0]
    key = l.split()[0]
    key = [key[l.find('[')+1:l.find(']')],]
    continue
  if is_number(l.strip()[:precision]): 
    lst.append(float(l.strip()[:precision]))
    if l.strip()[-2:] == ');':
      if int(key[0]) != len(lst):
        print 'key %s does not have the right amount of items.'\
            %(key[0])
      tempdict = {}
      tempdict = tempdict.fromkeys(key,lst)
      mydict.update(tempdict)
      #print tempdict
      #pdb.set_trace()
      lst = []

file_name = file_name[:-4]+'.dat'
fb = open(file_name,'wb')
pickle.dump(mydict,fb)
print 'Dictionary file saved to %s' % (file_name)
fb.close()
