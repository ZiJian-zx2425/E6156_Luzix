import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class ContactService {
  private apiUrl = 'http://localhost:5002/contact-doctor'; // Adjust as needed

  constructor(private http: HttpClient) { }

  postMessage(googleId: string, doctorId: number, message: string) {
    const payload = {
      google_id: googleId,
      doctor_id: doctorId,
      message: message
    };
    return this.http.post(this.apiUrl, payload);
  }

  /*
  // In your contact.service.ts
  postMessage(googleId: string, doctorId: number, doctorName: string, message: string) {
    const payload = {
      google_id: googleId,
      doctor_id: doctorId,
      // doctor_name: doctorName, // Make sure to include this in the payload
      message: message
    };
    return this.http.post(this.apiUrl, payload);
  }*/


  deleteMessage(messageId: number) {
    return this.http.delete(`${this.apiUrl}/${messageId}`);
  }

  updateMessage(messageId: number, message: string) {
    console.log("Updating message with message id: " + messageId)
    return this.http.put(`${this.apiUrl}/${messageId}`, { message: message });
  }

  getMessages(googleId: string) {
    return this.http.get(`${this.apiUrl}/${googleId}`);
  }

}
