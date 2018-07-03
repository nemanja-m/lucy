import calendar
import datetime


def time():
    current_time = datetime.datetime.now().time().replace(microsecond=0)
    return str(current_time)


def date():
    current_date = datetime.datetime.now().date()
    return str(current_date)


def day():
    weekday = datetime.datetime.now().date().weekday()
    current_day = calendar.day_name[weekday]
    return current_day


def month():
    month = datetime.datetime.now().date().month
    current_month = calendar.month_name[month]
    return current_month


def year():
    return datetime.datetime.now().date().year


processing_methods = {
    '#time': time,
    '#date': date,
    '#day': day,
    '#month': month,
    '#year': year,
}


def postprocess(query):
    """Handles special queries and provides an adequate response.

    Special queries suck as 'what is the time?' needs real time data which can't
    be obtained from dataset. This method handles such queries to provide
    correct response. Special queries start with # sign.

    It handles following special queries related to date or time:
        time,
        date,
        day,
        month,
        year

    Args:
        query (str): Special query string.

    Returns:
        String response.
    """
    method = processing_methods.get(query)

    # When no postprocessing methods can be applied, return original query
    if method is None:
        return query

    return method()
