class Debugger:
    def __init__(self):
        self.print_counter = 0
    
    def printd(self, data):
        self.print_counter += 1
        print('##############################################')
        print('Debug {} : '.format(self.print_counter))
        print(data)
        print('##############################################')