def tokenize(expr: str):
    tokens = []
    current = ""
    for ch in expr:
        if ch in "() \n\t":
            if current:
                tokens.append(current)
                current = ""
            if ch in "()":
                tokens.append(ch)
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens

def parse_ast(tokens):
    if not tokens:
        return None, []
    t = tokens.pop(0)
    if t == '(':
        lst = []
        while tokens and tokens[0] != ')':
            item, tokens = parse_ast(tokens)
            lst.append(item)
        if tokens:
            tokens.pop(0)
        return lst, tokens
    elif t == ')':
        raise ValueError("Unexpected ')'")
    else:
        return t, tokens

atom = '(= (CanAccess  ) (, (RequiresClearance  ) (HasClearance  )))'
t = tokenize(atom)
print("TOKENS:", t)
ast, _ = parse_ast(t)
print("AST:", ast)
