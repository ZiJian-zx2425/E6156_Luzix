import {Component, OnInit} from '@angular/core';
import {ApiService} from "../api.service";  // Import ApiService

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit {
  question: string = '';
  messages: { text: string, isUser: boolean }[] = [];
  brainId = ''; // This will hold the brain ID once created

  constructor(private apiService: ApiService) {}  // Inject ApiService

  ngOnInit() {
    this.createBrain();
  }

  // Call this method somewhere in your component to create a new brain
  createBrain() {
    this.apiService.createBrain().subscribe(response => {
      this.brainId = response.id; // Replace with actual property name from response
    }, error => {
      console.error('Error creating brain:', error);
    });
  }

  askQuestion(): void {
    if (!this.question.trim()) return;

    // Add user question to messages
    this.messages.push({ text: this.question, isUser: true });

    // Use ApiService to send the question
    this.apiService.askQuestion(this.question, this.brainId).subscribe({
      next: (response) => {
        console.log("Received response:", response);  // Debug log
        // Add AI response to messages
        this.messages.push({ text: response.answer, isUser: false });  // Adjust depending on Quivr's response structure
        this.question = '';  // Reset the question input
      },
      error: (error) => {
        console.error('Error asking question:', error);
        this.messages.push({ text: 'Failed to get a response from the brain.', isUser: false });
      }
    });
  }
}
