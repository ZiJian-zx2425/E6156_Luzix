import { Injectable } from '@angular/core';
import {HttpClient, HttpParams} from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PatientsRecordsService {
  private apiUrl = 'http://localhost:5001/patients-records'; // URL to your Flask API
  private doctorsPatientsUrl = 'http://localhost:5001/doctors-patients'; // URL to your Flask API
  private searchbypatientUrl = 'http://localhost:5001/search-by-patient'; // URL to your Flask API
  constructor(private http: HttpClient) {}

  getPatients(doctorId: string): Observable<string[]> {
    return this.http.get<string[]>(`${this.doctorsPatientsUrl}?doctor_id=${doctorId}`);
  }

  getPatientRecords(patientNames: string[]): Observable<any[]> {
    // Construct query parameters from the patient names array
    let params = new HttpParams();
    for (let name of patientNames) {
      params = params.append('patient_names[]', name);
    }
    return this.http.get<any[]>(this.apiUrl, { params });
  }

  searchByPatientName(name: string): Observable<any[]> {
    const params = new HttpParams().set('name', name);
    return this.http.get<any[]>(this.searchbypatientUrl, { params });
  }


}
