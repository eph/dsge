def write_latex_model_prior(mod, prior_list_or_dict, ncols=2, replace_function_or_dict=None):

    if type(prior_list_or_dict) == list:
        prior_dict = {'': prior_list_or_dict}
    else:
        prior_dict = prior_list_or_dict

    if replace_function_or_dict == None:
        replace = lambda x: x
    elif type(replace_function_or_dict) == dict:
        replace = lambda x: replace_function_or_dict[x]
    else:
        replace = replace_function_or_dict

    all_prior_parameters = mod['__data__']['estimation']['prior']

    header_str = r"""\begin{table}[H]
    \caption{\textsc{Prior Distribution}}
    \begin{center}
    \vspace*{1cm}
    \begin{tabular}{%s} \hline\hline
    %s \\  \hline 
    """ % (ncols*'llcc',
           ' & '.join(ncols*['Name & Density & Para (1) & Para (2)']))

    footer_str = r"""\hline\end{tabular}\end{center}
{\it Notes:} Para (1) and Para (2) correspond to the mean and standard deviation of the
Beta, Gamma, and Normal distributions and to the upper and lower bounds of the support
for the Uniform distribution.  For the Inv. Gamma distribution, Para (1) and Para (2) refer to
$s$ and $\nu$, where $p(\sigma|\nu, s)\propto \sigma^{-\nu-1}e^{-\nu s^2/2\sigma^2}$.
\end{table}"""


    rows = []
    for prior_group_header, prior_group in prior_dict.items():

        if len(prior_group_header) > 0:
            rows.append('   & \multicolumn{%d}{c}{%s} \\' % (3*ncols, prior_group_header))

        for para_row in zip(*[iter(prior_group)]*ncols):
            row = []
            if type(para_row)==str: para_row = [para_row]
            for para in para_row:
                dens, p1, p2 = all_prior_parameters[para]
                row.append('%s & %s & %5.2f & %5.2f' % (replace(para),
                                                        dens.replace('_','. ').title(),
                                                        p1, p2))


            rows.append(' & '.join(row) + r' \\')

    table_string = header_str + '\n'.join(rows) + '\n' + footer_str
    return table_string
