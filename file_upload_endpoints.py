@app.route('/api/request/<int:request_id>/send', methods=['POST'])
@login_required
def send_message(request_id):
    req = ServiceRequest.query.get_or_404(request_id)
    
    # Authorization: Farmer (owner) or CS (assigned) can send messages
    if current_user.role == 'farmer' and req.farmer_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    if current_user.role == 'cs' and req.cs_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    
    data = request.get_json()
    message_text = data.get('message', '').strip() if data else ''
    
    # Handle file upload if present
    file_path = None
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename and allowed_file(file.filename):
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                return {'error': 'File too large. Maximum size is 5MB.'}, 400
            
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
    
    # Reopening logic: if closed, reopen to pending
    if req.status == 'closed':
        req.status = 'pending'
        req.cs_id = None
        req.closed_at = None
        db.session.commit()
        print(f"DEBUG: Request {request_id} reopened by {current_user.username}")
    
    # Create message
    msg = ChatMessage(
        request_id=request_id,
        sender_id=current_user.id,
        message=message_text if message_text else None,
        file_path=file_path
    )
    db.session.add(msg)
    db.session.commit()
    
    return {'status': 'sent', 'message_id': msg.id}

@app.route('/uploads/<filename>')
@login_required
def download_file(filename):
    """Serve uploaded files with permission check"""
    # Get the file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Find the message containing this file
    msg = ChatMessage.query.filter(ChatMessage.file_path.contains(filename)).first()
    if not msg:
        return {'error': 'File not found'}, 404
    
    req = msg.request
    
    # Check permissions: User must be involved in the request or be admin
    if current_user.role == 'admin':
        pass  # Admin can access all files
    elif current_user.role == 'farmer' and req.farmer_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    elif current_user.role == 'cs' and req.cs_id != current_user.id:
        return {'error': 'Unauthorized'}, 403
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
