import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { AppRoutingModule } from './app-routing.module'; // Import AppRoutingModule here
import { AppComponent } from './app.component';
import { ChatComponent } from './chat/chat.component';
import { AuthSuccessComponent } from './auth-success/auth-success.component';
import { NavbarComponent } from './navbar/navbar.component';
import { ContactDoctorComponent } from './contact-doctor/contact-doctor.component';

@NgModule({
  declarations: [
    AppComponent,
    ChatComponent,
    AuthSuccessComponent,
    NavbarComponent,
    ContactDoctorComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    AppRoutingModule, // Include it in the imports array
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
