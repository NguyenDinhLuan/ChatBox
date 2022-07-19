from django import template
register = template.Library()

@register.filter(name="upper")
def do_upper(value):
    return value.upper()

@register.filter(name="removespace")
def do_removespace(value):
    return value.lower().replace(" ", "_")

@register.filter(name="captitalize")
def do_captitalize(value):
    return value.title()

