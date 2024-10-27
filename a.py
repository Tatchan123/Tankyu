import matplotlib.pyplot as pyp
import numpy as np
import openpyxl
wb = openpyxl.load_workbook('result.xlsx')
sheet = wb['Sheet1']
sheet.cell(row=5, column=1).value = "aaaaa"
wb.save('./Excelサンプル.xlsx')
wb.close()
