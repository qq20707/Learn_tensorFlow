import pygal
from random import randint 
bar_chart = pygal.Bar()
bar_chart.title="Remarquable sequences"
bar_chart.x_labels = map(str,range(11))
bar_chart.add('Fibonacci',[0,1,1,2,3,5,8,13,21,34,55,89])
bar_chart.add('Padovan',[1,1,1,2,2,3,4,5,7,9,12])
bar_chart.render_to_file('bar_char.svg')
#bar_chart.render()
print(map(str,range(11)))

class Die():
	"""docstring for Die"""
	def __init__(self, num_side=6):
		super(Die, self).__init__()
		self.num_side=num_side
		
	def roll(self):
		return randint(1,self.num_side)
		