#! /usr/bin/python

import os

#在文件的指定位置插入内容
def writeCode2File(file_path, line_index, code, head):
    fp = open(file_path)
    lines = []
    print("writeCode2File code:", code)
    # 循环遍历文件内容
    for i, line in enumerate(fp):
        if line_index == i :
            if head:
                lines.append(str(code) + str(line))
            else:
                lines.append(str(line).replace('\n',' ') + str(code))
        else:
            lines.append(line)
    # # 传入的文件位置暂无内容 则直接拼入一行内容
    # if len(fp.readlines()) < line_index:
    #     lines.append(code)
    fp.close()

    s = "".join('%s' % a for a in lines)
    fp = open(file_path, 'w')
    fp.write(s)
    fp.close()

#在文件的指定位置插入内容
def writeCode2FileAndRemoveN(file_path, line_index, code, head):
    fp = open(file_path)
    lines = []
    code = str(code).replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
    # 循环遍历文件内容
    for i, line in enumerate(fp):
        line = str(line).replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
        if line_index == i :
            if head:
                lines.append(str(code) + str(line))
            else:
                lines.append(str(line) + str(code))
        else:
            lines.append(line)
    # 传入的文件位置暂无内容 则直接拼如一行内容
    if len(fp.readlines()) <= line_index:
        lines.append(code)
    fp.close()

    s = "".join('%s' % a for a in lines)
    fp = open(file_path, 'w')
    fp.write(s)
    fp.close()

#在文件的指定位置插入内容
def copyFile(source_file_path, target_file_path):
    fp = open(source_file_path)
    lines = []
    # 循环遍历文件内容
    for i, line in enumerate(fp):
        lines.append(line)
    fp.close()

    s = ''.join(lines)
    fp = open(target_file_path, 'w')
    fp.write(s)
    fp.close()

#在文件的指定位置插入内容
def getLineFromFile(file_path, line_index):
    fp = open(file_path)
    lines = []
    # 循环遍历文件内容
    for i, line in enumerate(fp):
        lines.append(line + '\n')
        if line_index == i :
            lines.append(line)
            fp.close()
            return ''.join(lines)
    fp.close()
    return ''

#在文件的指定位置插入内容
def getOneLineFromFile(file_path, line_index):
    fp = open(file_path)
    lines = []
    # 循环遍历文件内容
    for i, line in enumerate(fp):
        if line_index == i :
            lines.append(line)
            fp.close()
            return ''.join(lines)
    fp.close()
    return ''

def writeTryCatchBlock2File(file_path, begin_index, end_index, catch_code):
    #插入try块
    writeCode2File(file_path, begin_index, 'try{\n', True)
    # 插入catch块
    writeCode2File(file_path, end_index+1, catch_code, True)

def writeTryBlock2File(file_path, begin_index, end_index):
    writeCode2File(file_path, begin_index, 'try { ', True)
    writeCode2File(file_path, end_index, ' }\n', False)

def writeCatchBlock2File(file_path, end_index, catch_code):
    # 插入catch块
    writeCode2File(file_path, end_index+1, catch_code, True)



def testEditFile():
    catch_code = 'catch(Exception e){ \n\tlog.error(e);\n\treturn;\n}'
    print("writing")
    #对第13行-15行进行try-catch
    writeTryCatchBlock2File('ImallApplication.java', 13, 15, catch_code)
    print("success")

# testEditFile()