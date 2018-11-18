from backup import *


s = [1, 2, 3, 4, 5]
fName = backupSave(s,"test")
print(fName)
print(backupLoad(fName))