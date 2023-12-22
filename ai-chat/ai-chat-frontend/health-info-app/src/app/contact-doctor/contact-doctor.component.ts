import {Component, OnInit} from '@angular/core';
import {AuthService} from "../services/auth/auth.service";
import {ContactService} from "../contact.service";

@Component({
  selector: 'app-contact-doctor',
  templateUrl: './contact-doctor.component.html',
  styleUrls: ['./contact-doctor.component.css']
})
export class ContactDoctorComponent implements OnInit {
  selectedDoctorId: number | null = null; // If you have a default selection, use that ID
  messageText: string = ''; // Initialize to an empty string
  messages: any[] = [];
  googleId: string = '';

  editingMessage: any = null; // Stores the message currently being edited
  editingMessageId: number | null = null;
  editingText: string = '';


  doctors = [
    { id: 1, name: 'Dr. Smith' },
    { id: 2, name: 'Dr. Jones' }
    // Add more doctors as needed
  ];

  constructor(private authService: AuthService, private contactService: ContactService) { }

  ngOnInit(): void {
    this.loadDoctors();
    this.loadMessages();
    this.googleId = this.authService.getUserGoogleId();
    console.log(this.googleId) // add dummy google id to database? no, don't need
  }

  loadDoctors() {
    // Assuming you have a method to fetch doctors, after fetching:
    if (this.doctors.length > 0) {
      this.selectedDoctorId = this.doctors[0].id; // Default to the first doctor's ID
    }
  }

  loadMessages() {
    console.log("loading messages from backend")
    this.contactService.getMessages(this.googleId).subscribe(data => {
      this.messages = data as any[];
    });
  }

  

  /*
  // Example method in your Angular component that fetches messages
  loadMessages(): void {
    // Assuming this method fetches the messages and stores them in this.messages
    this.contactService.getMessages(this.googleId).subscribe((messages) => {
      this.messages = messages.map((message) => {
        // Find the doctor's name using the doctor_id from the message
        const doctor = this.doctors.find((d) => d.id === message.doctor_id);
        return {
          ...message,
          doctor_name: doctor ? doctor.name : 'Unknown' // Add the doctor's name to the message object
        };
      });
    });
  }*/


  postMessage(selectedDoctorId: number | null, messageText: string) {
    if (this.selectedDoctorId && this.messageText) {
      this.contactService.postMessage(this.googleId, this.selectedDoctorId, this.messageText).subscribe(() => {
        this.loadMessages();
        this.messageText = ''; // Clear the message input after sending
      });
    } else {
      // Handle case where no doctor is selected
      console.error('No doctor selected');
    }
  }


  getDoctorNameById(doctorId: number): string {
    const doctor = this.doctors.find(d => d.id === doctorId);
    return doctor ? doctor.name : 'Unknown Doctor';
  }

  deleteMessage(messageId: number) {
    console.log(this.googleId) // add dummy google id to database? no, don't need
    this.contactService.deleteMessage(messageId).subscribe(() => {
      this.loadMessages();
    });
  }

  startEditMessage(message: any): void {
    this.editingMessage = { ...message };
    this.editingMessageId = message.id; // Set the ID of the message being edited
    this.messageText = message.message; // Copy the message text to a variable bound to the textarea
  }

  submitEditMessage(): void {
    console.log(this.googleId) // add dummy google id to database? no, don't need
    console.log("startEditMessage with editingMessageId: " + this.editingMessage) //null
    if (this.editingMessage) {
      this.contactService.updateMessage(this.editingMessage.id, this.editingMessage.message).subscribe(() => {
        // Replace the old message with the new one in the messages array
        console.log("contactService.updateMessage called")
        const index = this.messages.findIndex(m => m.id === this.editingMessage.id);
        if (index !== -1) {
          this.messages[index] = this.editingMessage;
        }
        // Clear the editing message
        this.editingMessage = null;
        // Possibly re-fetch messages or update the UI as needed
      });
    }
  }

  cancelEdit(): void {
    console.log(this.googleId) // add dummy google id to database? no, don't need
    // Clear the editing message
    this.editingMessage = null;
  }

}
