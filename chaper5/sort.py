class SQList():
	def __init__(self,lis=None):
		self.r=lis

	#shell sort
	def ShellSort(self):
		lis = self.r
		length = len(lis)
		k=int(length//2)

		while k>0:
			for i in range(k,length):
				if lis[i]<lis[i-k]:
					temp = lis[i]
					j = i-k
					while j>=0 and temp<lis[j]:
						lis[j+k]=lis[j]
						j=j-k
					lis[j+k]=temp

			k = k//2
		return lis

		#Insert sort
	def InsertSort(self):
		lis = self.r
		length = len(lis)
		for i in range(1,length):
			if lis[i]<lis[i-1]:
				temp = lis[i]
				j = i-1
				while j>=0 and temp < lis[j]:
					lis[j+1]=lis[j]
					j=j-1

				lis[j+1]=temp 

		return lis

if __name__ == '__main__':

	sortbeforelis =[3,41,53,123,13,8,7,11,23,90,1]
	print('Shell Sort:')
	print(str(sortbeforelis))
	alis = []
	sqlist = SQList(sortbeforelis)
	alis = sqlist.ShellSort()					
	print(alis)

	sortbeforelis =[33,41,53,123,13,8,7,11,23,90,21]
	print('\n Insert Sort:')
	print(sortbeforelis)
	sqlist = SQList(sortbeforelis)
	ali= sqlist.InsertSort()
	print(ali)
