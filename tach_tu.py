from pyvi import ViTokenizer, ViPosTagger

# return "Trường đại_học bách_khoa hà_nội"
print(ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội"))

#return (['Trường', 'đại_học', 'Bách_Khoa', 'Hà_Nội'], ['N', 'N', 'Np', 'Np'])
print(ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội")))

from pyvi import ViUtils
# remove dấu 
ViUtils.remove_accents(u"Trường đại học bách khoa hà nội")

from pyvi import ViUtils
# thêm dấu
ViUtils.add_accents(u'truong dai hoc bach khoa ha noi')