

j=''' 
(list (description Vermont) (description Connecticut) (description \"New Hampshire\") (description Massachusetts))
'''
end = str(j).index(')')
start = str(j).index('description ')+len('description ')
print(j[start:end])


