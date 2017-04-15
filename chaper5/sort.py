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



if __name__ == '__main__':

	sortbeforelis =[3,41,53,123,13,8,7,11,23,90,1]
	alis = []
	sqlist = SQList(sortbeforelis)
	alis = sqlist.ShellSort()
	print(sortbeforelis)					
	print(alis)