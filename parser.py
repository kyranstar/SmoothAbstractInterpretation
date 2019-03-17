from lark import Lark

l = Lark(''' start: "double" NAME "(" arglist ")" "{" stmt_block "}"
             arg: "double" NAME
             arglist: (arg ",")* arg | 

            %import common.WORD   // imports from terminal library
            %ignore " "           // Disregard spaces in text
         ''')

def parse(prog):
    ast = l.parse(prog)
    return traverse(ast)

def traverse(ast):
    if ast.data == 'program':
        _, _, _, arglist, _, _, stmt_block, _ = ast.children 
    else:
        raise SyntaxError('Unknown instruction: %s' % ast.data) 
