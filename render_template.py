import jinja2

def m_gen(filename="bomze.txt"):
    handle = open(filename)
    for line in handle:
        a,b,c,d,e,f,g,h,i = map(int, line.split())
        yield """\\left(\\begin{matrix}
        %s & %s & %s \\\\
        %s & %s & %s \\\\
        %s & %s & %s
        \\end{matrix} \\right)""" % (a,b,c,d,e,f,g,h,i)

template_loader = jinja2.FileSystemLoader("./")
template_env = jinja2.Environment(loader=template_loader)
template = template_env.get_template("template.html")

#print list(enumerate(m_gen()))[0]

output = template.render(ms=list(enumerate(m_gen())))
f = open("plots.html",'w')
f.write(output)