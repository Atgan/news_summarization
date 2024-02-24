




def response_return(success, data=None, message=None):
    response = {
        'success': success,
        'data': data,
        'message': message
    }
    if success:
        del response['message']
    else:
        del response['data']
        
    return response