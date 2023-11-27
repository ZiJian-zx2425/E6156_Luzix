import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import { Observable } from 'rxjs'

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly apiUrl = 'http://127.0.0.1:5000';
  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
      'Accept': 'application/json'  // This line is sometimes necessary
    })
  };

  constructor(private http: HttpClient) {}

  createBrain(): Observable<any> {
    return this.http.post(`${this.apiUrl}/create-brain`, {}, this.httpOptions);
  }

  askQuestion(question: string, brainId: string): Observable<any> {
    const endpoint = `${this.apiUrl}/ask/${brainId}`;
    const body = { question };
    return this.http.post(endpoint, body);
  }
}
