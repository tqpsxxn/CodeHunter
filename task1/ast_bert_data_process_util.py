import logging
import re

import javalang

logging.basicConfig(filename='new_data_process.log', level=logging.DEBUG)

parse_error_count = 0

PAD = 'PAD'

# 获取经过语法树解析之后的标记 输入为code_lines,tags 输出为经过处理后的 code_lines,tags
# 1、由于数据集中的代码存在不规范或是由于单行代码过长导致了分行，因此我们借助语法树解析，把本属于一行代码的多行代码，重新组合成一行代码
# 2、数据集中部分java函数无法被解析，这部分数据被过滤了
# 3、只要能够成功解析成功，就会在每一行代码的最前面放上对这一行代码的语法树解析结果，用于增强特征!
def get_feature_tokens(code_lines, tags):
    # 原始的代码
    old_lines = code_lines
    # 去除空格 格式化处理之后的代码
    code_lines = [re.sub(r"\s*([().])\s*", r"\1", line) for line in code_lines]
    code_lines = [line.replace("\r", " ") for line in code_lines]
    code_lines = [line.replace("\t", " ") for line in code_lines]


    feature_tokens = []
    new_tags  = []
    new_lines = []
    try:
        # tree = javalang.parse.parse("public void test(){log.info("")};")
        codestr = '\n'.join(code_lines)
        programtokens = javalang.tokenizer.tokenize(codestr)
        # print("programtokens",list(programtokens))
        parser = javalang.parse.Parser(programtokens)
        programast = parser.parse_member_declaration()
        # 递归处理，获取每一行的标记
        feature_tokens = parse_feature_tokens(programast)
        # 加上最后一行标识（方法的反括号：'}'）
        feature_tokens.update({len(code_lines): ['MethodDeclaration', 'END']})
        sorted_keys = sorted(feature_tokens.keys())
        for i in range(len(sorted_keys)):
            # 行号 获取范围，如果多行代码在经过ast解析之后在同一行，那么需要进行合并
            line_number = sorted_keys[i]
            line_number_end = line_number + 1
            if(i < len(sorted_keys) - 1):
                line_number_end = sorted_keys[i + 1]
            new_line = ''
            # line_number下标从1开始，因此遍历时需要-1
            for line_no in range(line_number - 1, line_number_end - 1):
                new_line = new_line + old_lines[line_no]
            new_line = new_line + ' '.join(feature_tokens.get(line_number))
            new_tags.append(tags[line_number - 1])
            new_lines.append(new_line)
        return new_lines,new_tags
    except Exception as e:
        # logging.error("get_feature_tokens error")
        pass
    return new_lines,new_tags

#严格保持原来的行号
def get_feature_tokens_for_api(code_lines, tags):
    # 原始的代码
    old_lines = code_lines
    # 去除空格 格式化处理之后的代码
    code_lines = [re.sub(r"\s*([().])\s*", r"\1", line) for line in code_lines]
    code_lines = [line.replace("\r", " ") for line in code_lines]
    code_lines = [line.replace("\t", " ") for line in code_lines]


    feature_tokens = []
    new_tags  = []
    new_lines = []
    try:
        # tree = javalang.parse.parse("public void test(){log.info("")};")
        codestr = '\n'.join(code_lines)
        programtokens = javalang.tokenizer.tokenize(codestr)
        # print("programtokens",list(programtokens))
        parser = javalang.parse.Parser(programtokens)
        programast = parser.parse_member_declaration()
        # 递归处理，获取每一行的标记
        feature_tokens = parse_feature_tokens(programast)
        # 加上最后一行标识（方法的反括号：'}'）
        feature_tokens.update({len(code_lines): ['MethodDeclaration', 'END']})
        for i in range(len(code_lines)):
            # 行号 获取范围，如果多行代码在经过ast解析之后在同一行，那么需要进行合并
            new_line = ''
            new_line = new_line + old_lines[i]
            if(feature_tokens.get(i + 1) != None):
                feature_token = ' '.join(feature_tokens.get(i + 1))
                new_line = new_line + feature_token
            new_tags.append(tags[i])
            new_lines.append(new_line)
        return new_lines,new_tags
    except Exception as e:
        # logging.error("get_feature_tokens error")
        pass
    return new_lines,new_tags


# 递归读取语法树结构，构造出 {1 : "ast info"} 格式数据，其中1表示代码的第几行，下标从1开始
def parse_feature_tokens(programast):
    res = {}
    try:
        if (programast == None):
            return res
        # 获取类型
        code_type = type(programast)
        code_type = str(code_type.__name__).split('.')[-1]

        if (code_type == 'tuple' or code_type == 'list'):
            # for p in programast:
            #     res.update(parse_feature_tokens(p))
            return res

        now = []
        if (isinstance(programast, javalang.tree.MethodDeclaration)):
            now.append('START')

        if (programast.position != None):
            # 获取行号
            line_number = programast.position[0]
            now = [code_type]
            # 当前行
            now = {line_number: now}
            res.update(now)
        # 获取修饰符
        # modifiers = []
        # if hasattr(programast, 'modifiers'):
        #     modifiers = [m for m in programast.modifiers]
        #     now.extend(modifiers)

        # 方法定义
        if (hasattr(programast, 'body') and programast.body != None):
            if(isinstance(programast.body, list)):
                for i in programast.body:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.body))

        if (hasattr(programast, 'statements') and programast.statements != None):
            if (isinstance(programast.statements, list)):
                for i in programast.statements:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.statements))

        if (hasattr(programast, 'then_statement') and programast.then_statement != None):
            if (isinstance(programast.then_statement, list)):
                for i in programast.then_statement:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.then_statement))
            # for i in programast.then_statement:
            #     res.update(parse_feature_tokens(i))

        if (hasattr(programast, 'else_statement') and programast.else_statement != None):
            if (isinstance(programast.else_statement, list)):
                for i in programast.else_statement:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.else_statement))
            # res.update(parse_feature_tokens(programast.else_statement))
            # for i in programast.else_statement:
            #     res.update(parse_feature_tokens(i))

        if (hasattr(programast, 'block') and programast.block != None):
            if (isinstance(programast.block, list)):
                for i in programast.block:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.block))

        if (hasattr(programast, 'catches') and programast.catches != None):
            if (isinstance(programast.catches, list)):
                for i in programast.catches:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.catches))


        if (hasattr(programast, 'finally_block') and programast.finally_block != None):
            if (isinstance(programast.finally_block, list)):
                for i in programast.finally_block:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.finally_block))


        if (hasattr(programast, 'cases') and programast.cases != None):
            if (isinstance(programast.cases, list)):
                for i in programast.cases:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.cases))

        if (hasattr(programast, 'expression') and programast.expression != None):
            if (isinstance(programast.expression, list)):
                for i in programast.expression:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.expression))
    except Exception as e:
        logging.error("parse_feature_tokens error", programast, e)
    return res