import matplotlib.pyplot as pyp
import numpy as np
import openpyxl
import os
file = os.path.abspath("C:\\Users\\Owner\\Documents\\Tankyu_Shujiro\\Local\\Tankyu\\result.xlsx")
wb = openpyxl.load_workbook(file)
sheet = wb['Sheet1']
sheet.cell(row=5, column=1).value = "aaaaa"
wb.save('./Excelサンプル.xlsx')
wb.close()
