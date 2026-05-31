from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import urllib.parse

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

note_tags = db.Table('note_tags',
    db.Column('note_id', db.Integer, db.ForeignKey('note.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat()
        }

class NoteVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    note_id = db.Column(db.Integer, db.ForeignKey('note.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    version_number = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'note_id': self.note_id,
            'title': self.title,
            'content': self.content,
            'version_number': self.version_number,
            'created_at': self.created_at.isoformat()
        }

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = db.Column(db.Boolean, default=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    version_count = db.Column(db.Integer, default=0)
    
    tags = db.relationship('Tag', secondary=note_tags, backref=db.backref('notes', lazy='dynamic'))
    versions = db.relationship('NoteVersion', backref='note', lazy='dynamic', order_by='NoteVersion.version_number.desc()')

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': [tag.name for tag in self.tags],
            'version_count': self.version_count
        }

with app.app_context():
    db.create_all()

def get_or_create_tag(tag_name):
    tag = Tag.query.filter_by(name=tag_name).first()
    if not tag:
        tag = Tag(name=tag_name)
        db.session.add(tag)
        db.session.flush()
    return tag

def save_version(note):
    note.version_count += 1
    version = NoteVersion(
        note_id=note.id,
        title=note.title,
        content=note.content,
        version_number=note.version_count
    )
    db.session.add(version)

@app.route('/notes', methods=['POST'])
def create_note():
    data = request.get_json()
    if not data or 'title' not in data or 'content' not in data:
        return jsonify({'error': 'Title and content are required'}), 400
    
    note = Note(title=data['title'], content=data['content'])
    db.session.add(note)
    db.session.flush()
    
    if 'tags' in data and isinstance(data['tags'], list):
        for tag_name in data['tags']:
            tag = get_or_create_tag(tag_name.strip())
            note.tags.append(tag)
    
    save_version(note)
    db.session.commit()
    
    return jsonify(note.to_dict()), 201

@app.route('/notes', methods=['GET'])
def get_notes():
    search = request.args.get('search', '')
    tag = request.args.get('tag', '')
    query = Note.query.filter(Note.is_deleted == False)
    
    if search:
        query = query.filter(
            Note.title.contains(search) | Note.content.contains(search)
        )
    
    if tag:
        query = query.filter(Note.tags.any(name=tag))
    
    notes = query.order_by(Note.updated_at.desc()).all()
    return jsonify([note.to_dict() for note in notes])

@app.route('/notes/<int:note_id>', methods=['GET'])
def get_note(note_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    return jsonify(note.to_dict())

@app.route('/notes/<int:note_id>', methods=['PUT'])
def update_note(note_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    if 'title' in data:
        note.title = data['title']
    if 'content' in data:
        note.content = data['content']
    
    if 'tags' in data and isinstance(data['tags'], list):
        note.tags = []
        for tag_name in data['tags']:
            tag = get_or_create_tag(tag_name.strip())
            note.tags.append(tag)
    
    save_version(note)
    db.session.commit()
    return jsonify(note.to_dict())

@app.route('/notes/<int:note_id>', methods=['DELETE'])
def delete_note(note_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    note.is_deleted = True
    note.deleted_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'message': 'Note deleted successfully'}), 200

@app.route('/notes/<int:note_id>/versions', methods=['GET'])
def get_note_versions(note_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    versions = note.versions.all()
    return jsonify([v.to_dict() for v in versions])

@app.route('/notes/<int:note_id>/versions/<int:version_id>', methods=['GET'])
def get_note_version_detail(note_id, version_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    version = NoteVersion.query.filter_by(id=version_id, note_id=note_id).first()
    if not version:
        return jsonify({'error': 'Version not found'}), 404
    
    return jsonify(version.to_dict())

@app.route('/notes/<int:note_id>/versions/<int:version_id>/restore', methods=['POST'])
def restore_version(note_id, version_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    version = NoteVersion.query.filter_by(id=version_id, note_id=note_id).first()
    if not version:
        return jsonify({'error': 'Version not found'}), 404
    
    note.title = version.title
    note.content = version.content
    save_version(note)
    db.session.commit()
    
    return jsonify({
        'message': 'Note restored to version ' + str(version.version_number),
        'note': note.to_dict()
    })

@app.route('/notes/<int:note_id>/export/markdown', methods=['GET'])
def export_markdown(note_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    md_content = f"# {note.title}\n\n"
    md_content += f"**创建时间**: {note.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_content += f"**更新时间**: {note.updated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if note.tags:
        md_content += f"**标签**: {', '.join([tag.name for tag in note.tags])}\n"
    md_content += f"\n---\n\n{note.content}\n"
    
    filename = urllib.parse.quote(f"{note.title}.md")
    response = make_response(md_content)
    response.headers['Content-Type'] = 'text/markdown; charset=utf-8'
    response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
    return response

@app.route('/notes/<int:note_id>/export/pdf', methods=['GET'])
def export_pdf(note_id):
    note = Note.query.filter_by(id=note_id, is_deleted=False).first()
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#666666'
    )
    
    story.append(Paragraph(note.title, title_style))
    story.append(Paragraph(f"创建时间: {note.created_at.strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    story.append(Paragraph(f"更新时间: {note.updated_at.strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    if note.tags:
        story.append(Paragraph(f"标签: {', '.join([tag.name for tag in note.tags])}", meta_style))
    
    story.append(Spacer(1, 0.5 * inch))
    
    content = note.content.replace('\n', '<br/>')
    story.append(Paragraph(content, styles['BodyText']))
    
    doc.build(story)
    
    filename = urllib.parse.quote(f"{note.title}.pdf")
    buffer.seek(0)
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
    return response

@app.route('/tags', methods=['GET'])
def get_all_tags():
    tags = Tag.query.all()
    return jsonify([tag.to_dict() for tag in tags])

if __name__ == '__main__':
    app.run(debug=True)
