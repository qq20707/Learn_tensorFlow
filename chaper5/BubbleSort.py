class SQlist():
	"""docstring for SQlist"""
	def __init__(self, lis):
		#super(SQlist, self).__init__()
		self.lis = lis
		

	def BubbleSort(self):
		lis = self.lis
		length = len(lis)
		for i in range(0,length-1):
			for j in range(0,length-1-i):
				if lis[j]>lis[j+1]:
					temp = lis[j]
					lis[j] = lis[j+1]
					lis[j+1] = temp

		return lis

if __name__ == '__main__':

	beforelis=[10,90,40,56,92,3,5,2,103]
	sqlist = SQlist(beforelis)
	print("before sort list:")
	print(beforelis)
	afterlist = []
	afterlist = sqlist.BubbleSort()
	print("After sort list :")
	print(afterlist)