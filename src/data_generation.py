import numpy as np
import random
import names
import datetime
from essential_generators import DocumentGenerator


def generate_training_sample():
    def append_random(buf, choices, exclude=[]):
        assert(len(choices) - len(exclude) > 0)
        # idx = random.randint(0, len(choices) - 1)
        idx = 0
        buf.append(choices[idx])
        return idx

    begin = datetime.datetime.now()
    # input = ("Email my colleagues that I can't come to the meeting tomorrow")
    # output = ("BEGIN_CALL email ARG recipient ARG colleagues ARG_NAME
    #            message ARG_VALUE I can't come to the meeting tomorrow END_CALL")

    #
    # Bulding the output message
    #

    query = []
    output = ["BEGIN_CALL"]
    commands = ["email", "remind", "calender"]

    remind_args = ["when", "", "body", "when"]

    cmd_idx = append_random(output, commands)
    if cmd_idx == 0:
        generate_email_sample(query, output)

    output.append("END_CALL")
    delta = datetime.datetime.now() - begin
    print(delta)
    return (" ".join(query), " ".join(output))
    
def generate_email_sample(query, output):
    gen = DocumentGenerator()
    email_args = ["--recipients", "--subject", "--body", "--when"]
    include = np.random.choice(1, len(email_args));
    include[0] = 1
    recipients = [];
    subject = ""
    body = ""
    when = ""

    # Recipients
    output.append("ARG")
    num_recipients = random.randint(1, 5)
    for i in range(0, num_recipients):
        # TODO(alexander): maybe not only provide name of random people
        entry = ""
        if random.randint(0, 1) == 0: entry = gen.email();
        else: entry = names.get_full_name();
        recipients.append(entry)
        output.append(entry)
        if i < num_recipients - 1:
            output.append("AND")

    # Email subject
    output.append("ARG")
    subject = gen.sentence()
    output.append(subject)

    # Email body
    output.append("ARG")
    body = gen.paragraph()
    output.append(body)

    # Genreate when
    output.append("ARG")
    now = datetime.datetime.now()
    when = now.strftime("%Y-%m-%d %H:%M:%S")
    output.append(when)
    
    inputs = " ".join(["email"])
            

